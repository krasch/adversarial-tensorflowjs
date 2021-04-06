// FGSM configuration; these values are cherry-picked
// (FGSM is the not strongest attack, and the selected flower mostly
//  turns into a vase, no matter what targetClass was selected)
const targetClass = 883; // class ID for vase
const epsilon = 5.0;   // strength of the perturbation

// loaded model will be stored here
let model = null;


// just a simple helper function that takes a prediction dictionary from tensorflow.js
// and formats it into a nice string
function formatPrediction(prediction){
    const roundedProbability = Math.round(prediction["probability"] * 100.0) / 100.0;
    return prediction["className"] +" (" + roundedProbability + ")";
}

// perform a targeted Fast Gradient Sign method attack
// trying to "turn" the original image into the given target class
function targetedFGSM(model, originalImage, targetClass, epsilon){
    // tf.grad expects a loss function that takes only one input: one (image) tensor
    // however, we want the cross-entropy loss with respect to an image AND the target class
    // we get around by specifying the loss function inside the outer function (closure)
    // this way the function has access to both the image as well as the target class
    function loss(image){
        let targetOneHot = tf.oneHot([targetClass], 1000);
        let logits = model.infer(image);
        return tf.losses.softmaxCrossEntropy(targetOneHot, logits);
    }

    // now we can initialise a function that calculates a gradient
    // given the specified loss function
    const calculateGradient = tf.grad(loss);

    // tf.tidy automatically disposes all tensors that are not needed anymore
    // (so we don't get memory leaks)
    return tf.tidy(() => {
            // let's get the gradient with respect to the original image
            const gradient = calculateGradient(originalImage);

            // apply the fast gradient sign method
            const perturbation = gradient.sign().mul(epsilon);
            let adversarial = tf.sub(originalImage, perturbation);

            // pixel values must be between [0, 255]
            adversarial = adversarial.clipByValue(0, 255);

            return adversarial;
        });
}

// this function is called when the "Go!" button is clicked
// any function that "awaits" asynchronous functions, must itself be marked as async
async function runAttack(){
    // if this is the first time the button was clicked, we need to load the model
    if (model === null)
        model = await mobilenet.load();

    // how much memory is tensorflow.js using before generating the adversarial image?
    const initialMemoryUsage = tf.memory().numBytes;

    // run the classifier on the original image
    // the result is an array with the Top3 predictions
    const originalElement = document.getElementById("original-image");
    const originalPredictions = await model.classify(originalElement);

    // lets write the highest-probable prediction onto the webpage
    const originalTextElement = document.getElementById("original-text");
    originalTextElement.innerHTML = formatPrediction(originalPredictions[0]);

    // to generate the adversarial,
    // we let tensorflow grab the image data from the <img> DOM element
    // and then run the targetedFGSM function
    const originalTensor = tf.browser.fromPixels(originalElement);
    const adversarialTensor = targetedFGSM(model, originalTensor, targetClass, epsilon);

    // display the adversarial image on the webpage
    // need to store the normalized tensor into a variable
    // so we can dispose it later (avoid memory leaks)
    const adversarialElement = document.getElementById("adversarial-image")
    const adversarialTensorNormalized = adversarialTensor.div(255);
    tf.browser.toPixels(adversarialTensorNormalized, adversarialElement);

    // run the classifier on the generated adversarial image
    const adversarialPredictions = await model.classify(adversarialTensor);

    // and again write the highest-probable prediction onto the webpage
    const adversarialTextElement = document.getElementById("adversarial-text");
    adversarialTextElement.innerHTML = formatPrediction(adversarialPredictions[0]);

    // clean up to avoid memory leaks
    originalTensor.dispose();
    adversarialTensor.dispose();
    adversarialTensorNormalized.dispose();

    // check if still have any memory leaks
    const leakingMemory = tf.memory().numBytes - initialMemoryUsage;
    console.log("Memory leakage: " + leakingMemory + " bytes");
}


