let cachedResults = {
    context: [],
    similarity: []
};

let activeRank = "context";

const sentenceInput = document.getElementById("inputA");
const lengthWarning = document.getElementById("err_length");

sentenceInput.addEventListener("input", () => {
    if (sentenceInput.value.length > 80) {
        lengthWarning.style.display = "block";
    } else {
        lengthWarning.style.display = "none";
    }
});


document.getElementById('two-input-form').addEventListener('submit', function (e) {
    e.preventDefault();

    const sentence = document.getElementById('inputA').value.trim();
    const word = document.getElementById('inputB').value.trim();
    const result = document.getElementById('result');
    
    // Error handling
    if (!sentence.includes(word)) {
        result.innerHTML = `
            <div class="muted" style="color:#b91c1c;">
                The target word does not appear in the input text.
            </div>
        `;          
        return;
    }

    if (!sentence || !word) {
        result.innerHTML = `
            <div class="muted" style="color:#b91c1c;">
                Please enter both values.
            </div>
        `;          
        return;
    }

    if (word.includes(" ")) {
        result.innerHTML = `
            <div class="muted" style="color:#b91c1c;">
                Please enter only one word for Target Word.
            </div>
        `;   
        return
    }

    if (sentence.lenghth > 120) {
        result.innerHTML += `
            <div class="muted" style="color:#b91c1c;">
                Larger Input Text may result in slower or more inconsistent results.
            </div>
        `;   
    }



    result.innerHTML = "<div class='muted'>Loading...</div>";

    // fetching logic for flask
    fetch("/get-synonyms", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({
            sentence: sentence,
            word: word,
            top_n: 10
        })
    })
    .then(res => res.json())
    .then(data => {
        cachedResults.context = data.context_rank;
        cachedResults.similarity = data.similarity_rank;

        renderResults(cachedResults[activeRank], word);
    })
    .catch(err => {
        result.innerHTML = "<span class='error'>Failed to analyze sentence</span>";
        console.error(err);
    });
});


function renderResults(list, word) {
    const result = document.getElementById("result");

    if (!list || list.length === 0) {
        result.innerHTML = "<div class='muted'>No results found.</div>";
        return;
    }

    const subtitle =
        activeRank === "context"
            ? "Scoring based on contextual fit using embeddings."
            : "Scoring based on similarity to original word post contextual fit";

    let html = `
        <div class="result-header">
            <strong>Best substitutions for "${word}"</strong>
            <div class="result-subtitle">${subtitle}</div>
        </div>
        <ul>
    `;

    list.forEach(item => {
        html += `
            <li>
                <strong>${item.word}</strong>
                <span class="muted">(${item.score.toFixed(3)})</span>
            </li>
        `;
    });

    html += "</ul>";
    result.innerHTML = html;
}


document.querySelectorAll(".tab").forEach(tab => {
    tab.addEventListener("click", () => {
        document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
        tab.classList.add("active");

        activeRank = tab.dataset.rank;
        renderResults(
            cachedResults[activeRank],
            document.getElementById("inputB").value
        );
    });
});
