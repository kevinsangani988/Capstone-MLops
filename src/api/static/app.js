const form = document.getElementById("predict-form");
const resultText = document.getElementById("result-text");
const probabilityText = document.getElementById("probability-text");
const predictButton = document.getElementById("predict-btn");

function setLoadingState(isLoading) {
  predictButton.disabled = isLoading;
  predictButton.textContent = isLoading ? "Predicting..." : "Run Prediction";
}

function collectPayload() {
  const formData = new FormData(form);
  const record = {};

  for (const [key, value] of formData.entries()) {
    const input = form.querySelector(`[name='${key}']`);
    if (input && input.type === "number") {
      record[key] = value === "" ? null : Number(value);
    } else {
      record[key] = value;
    }
  }

  return { instances: [record] };
}

async function runPrediction(event) {
  event.preventDefault();
  setLoadingState(true);
  resultText.textContent = "Running prediction...";
  probabilityText.textContent = "";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(collectPayload())
    });

    const data = await response.json();

    if (!response.ok) {
      const errorDetail = data?.detail || "Prediction failed. Please validate your inputs.";
      throw new Error(errorDetail);
    }

    const prediction = data.predictions?.[0];
    resultText.textContent = `Prediction: ${prediction}`;

    if (Array.isArray(data.probabilities) && data.probabilities.length > 0) {
      const probability = Number(data.probabilities[0]);
      if (!Number.isNaN(probability)) {
        const positiveClassLabel = data.positive_class_label || "Positive class";
        probabilityText.textContent = `Probability of ${positiveClassLabel}: ${(probability * 100).toFixed(2)}%`;
      }
    }
  } catch (error) {
    resultText.textContent = "Unable to get prediction.";
    probabilityText.textContent = error.message;
  } finally {
    setLoadingState(false);
  }
}

form.addEventListener("submit", runPrediction);
