async function analyzeComment() {
    const commentInput = document.getElementById("commentInput");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const loading = document.getElementById("loading");
    const resultSection = document.getElementById("resultSection");

    const text = commentInput.value.trim();

    if (!text) {
        alert("Please enter a comment first.");
        return;
    }

    analyzeBtn.disabled = true;
    loading.classList.remove("hidden");
    resultSection.classList.add("hidden");

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text,
                include_probabilities: true
            })
        });

        if (!response.ok) {
            throw new Error("Prediction request failed.");
        }

        const result = await response.json();

        document.getElementById("prediction").textContent = result.prediction;
        document.getElementById("confidence").textContent = result.confidence;
        document.getElementById("toxicityScore").textContent = result.toxicity_score;
        document.getElementById("severity").textContent = result.severity;
        document.getElementById("action").textContent = result.action;
        document.getElementById("reason").textContent = result.reason;

        resultSection.classList.remove("hidden");

    } catch (error) {
        console.error(error);
        alert("Unable to connect to the AI backend. Please try again.");
    } finally {
        analyzeBtn.disabled = false;
        loading.classList.add("hidden");
    }
}