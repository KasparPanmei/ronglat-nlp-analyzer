const analyzeButton = document.getElementById("analyze-button");
const input = document.getElementById("ronglat-input");
const errorBox = document.getElementById("error-message");
const loading = document.getElementById("loading-message");
const resultsSection = document.getElementById("results-section");

function escapeHTML(value) {
    return String(value ?? "").replace(/[&<>"']/g, (char) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
    })[char]);
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.hidden = false;
}

function setSummary(summary) {
    document.getElementById("summary-tokens").textContent = summary.tokens;
    document.getElementById("summary-known").textContent = summary.known;
    document.getElementById("summary-unseen").textContent = summary.unseen;
    document.getElementById("summary-entities").textContent = summary.entities;
    document.getElementById("summary-glossed").textContent = summary.glossed;
}

function renderTagging(results) {
    const body = document.getElementById("tagging-body");
    body.innerHTML = results.map((item) => `
        <tr>
            <td>${escapeHTML(item.token)}</td>
            <td>${item.gloss ? escapeHTML(item.gloss) : '<span class="muted-cell">Not found in corpus</span>'}</td>
            <td><span class="tag tag-pos">${escapeHTML(item.pos)}</span></td>
            <td><span class="tag tag-ner">${escapeHTML(item.ner)}</span></td>
            <td><span class="tag ${item.known ? "tag-known" : "tag-unseen"}">${item.known ? "Known" : "Unseen"}</span></td>
        </tr>
    `).join("");
}

function renderEntities(entities) {
    const target = document.getElementById("entities-list");
    if (!entities.length) {
        target.innerHTML = '<div class="empty-state">No named entities were identified in this input.</div>';
        return;
    }
    target.innerHTML = entities.map((entity) => `
        <div class="entity-row"><strong>${escapeHTML(entity.text)}</strong><span>${escapeHTML(entity.type)}</span></div>
    `).join("");
}

function renderGlosses(results) {
    const target = document.getElementById("glosses-list");
    const glossed = results.filter((item) => item.gloss);
    if (!glossed.length) {
        target.innerHTML = '<div class="empty-state">No matching English glosses were found in the corpus.</div>';
        return;
    }
    target.innerHTML = glossed.map((item) => `
        <div class="gloss-row"><strong>${escapeHTML(item.token)}</strong><span>${escapeHTML(item.gloss)}</span></div>
    `).join("");
}

function renderDistribution(distribution) {
    const target = document.getElementById("distribution-list");
    const entries = Object.entries(distribution).sort((a, b) => b[1] - a[1]);
    const max = Math.max(1, ...entries.map((entry) => entry[1]));
    target.innerHTML = entries.map(([label, count]) => `
        <div class="distribution-row">
            <span class="distribution-label">${escapeHTML(label)}</span>
            <div class="bar-track"><div class="bar-fill" style="width:${(count / max) * 100}%"></div></div>
            <span class="distribution-count">${count}</span>
        </div>
    `).join("");
}

function activateTabs() {
    document.querySelectorAll(".tab-button").forEach((button) => {
        button.addEventListener("click", () => {
            document.querySelectorAll(".tab-button").forEach((tab) => tab.classList.remove("active"));
            document.querySelectorAll(".tab-content").forEach((panel) => panel.classList.remove("active"));
            button.classList.add("active");
            document.getElementById(`tab-${button.dataset.tab}`).classList.add("active");
        });
    });
}

async function analyzeText() {
    const text = input.value.trim();
    errorBox.hidden = true;

    if (!text) {
        showError("Please enter Ronglat text before starting the analysis.");
        input.focus();
        return;
    }

    analyzeButton.disabled = true;
    loading.hidden = false;
    resultsSection.hidden = true;

    try {
        const response = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ text })
        });
        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "The analysis request failed.");
        }

        setSummary(data.summary);
        renderTagging(data.results);
        renderEntities(data.entities);
        renderGlosses(data.results);
        renderDistribution(data.pos_distribution);

        document.querySelectorAll(".tab-button").forEach((tab) => tab.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach((panel) => panel.classList.remove("active"));
        document.querySelector('[data-tab="tagging"]').classList.add("active");
        document.getElementById("tab-tagging").classList.add("active");

        resultsSection.hidden = false;
        resultsSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
        showError(error.message);
    } finally {
        loading.hidden = true;
        analyzeButton.disabled = false;
    }
}

analyzeButton.addEventListener("click", analyzeText);
input.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") analyzeText();
});
activateTabs();

const vocabularyBody =
    document.getElementById("vocabulary-body");

const vocabularySearch =
    document.getElementById("vocab-search");

const vocabularyPageSize =
    document.getElementById("vocab-page-size");

const vocabularyPrev =
    document.getElementById("vocab-prev");

const vocabularyNext =
    document.getElementById("vocab-next");

const vocabularyCount =
    document.getElementById("vocab-count");

const vocabularyPageLabel =
    document.getElementById("vocab-page-label");

const vocabularyError =
    document.getElementById("vocab-error");


let vocabularyPage = 1;

let vocabularyPages = 1;

let vocabularySearchTimer = null;


// Load vocabulary from Python API
async function loadVocabulary() {

    if (!vocabularyBody) return;

    vocabularyError.hidden = true;

    vocabularyBody.innerHTML = `
        <tr>
            <td colspan="3" class="vocab-placeholder">
                Loading vocabulary...
            </td>
        </tr>
    `;


    const params = new URLSearchParams({

        q: vocabularySearch.value.trim(),

        page: String(vocabularyPage),

        per_page: vocabularyPageSize.value

    });


    try {

        const response = await fetch(
            `/api/vocabulary?${params.toString()}`
        );

        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.error || "Could not load vocabulary."
            );

        }


        // Update statistics

        document.getElementById(
            "vocab-unique"
        ).textContent =
            Number(
                data.stats.unique_entries
            ).toLocaleString();


        document.getElementById(
            "vocab-rows"
        ).textContent =
            Number(
                data.stats.corpus_rows
            ).toLocaleString();


        document.getElementById(
            "vocab-glossed"
        ).textContent =
            Number(
                data.stats.entries_with_gloss
            ).toLocaleString();


        vocabularyPages =
            data.pagination.pages;

        vocabularyPage =
            data.pagination.page;


        // Render vocabulary rows

        if (!data.items.length) {

            vocabularyBody.innerHTML = `
                <tr>
                    <td colspan="3" class="vocab-placeholder">
                        No matching vocabulary entries found.
                    </td>
                </tr>
            `;

        } else {

            vocabularyBody.innerHTML =
                data.items.map(item => `

                    <tr>

                        <td class="vocab-token">
                            ${escapeHTML(item.token)}
                        </td>

                        <td>
                            ${item.gloss
                        ? escapeHTML(item.gloss)
                        : '<span class="muted-cell">No gloss recorded</span>'
                    }
                        </td>

                        <td>

                            <span class="frequency-pill">
                                ${Number(item.frequency).toLocaleString()}
                            </span>

                        </td>

                    </tr>

                `).join("");

        }


        // Update pagination information

        const first =
            data.pagination.total === 0
                ? 0
                : (
                    (vocabularyPage - 1) *
                    data.pagination.per_page
                ) + 1;


        const last = Math.min(

            vocabularyPage *
            data.pagination.per_page,

            data.pagination.total

        );


        vocabularyCount.textContent =

            `Showing ${first.toLocaleString()}–${last.toLocaleString()} ` +
            `of ${data.pagination.total.toLocaleString()} matching entries`;


        vocabularyPageLabel.textContent =

            `Page ${vocabularyPage} of ${vocabularyPages}`;


        vocabularyPrev.disabled =
            vocabularyPage <= 1;

        vocabularyNext.disabled =
            vocabularyPage >= vocabularyPages;


    } catch (error) {

        vocabularyBody.innerHTML = `
            <tr>
                <td colspan="3" class="vocab-placeholder">
                    Vocabulary could not be loaded.
                </td>
            </tr>
        `;

        vocabularyError.textContent =
            error.message;

        vocabularyError.hidden = false;

    }

}


// Search Ronglat words and English meanings
if (vocabularyBody) {

    vocabularySearch.addEventListener(
        "input",
        () => {

            clearTimeout(vocabularySearchTimer);

            vocabularySearchTimer = setTimeout(
                () => {

                    vocabularyPage = 1;

                    loadVocabulary();

                },
                250
            );

        }
    );


    // Change page size
    vocabularyPageSize.addEventListener(
        "change",
        () => {

            vocabularyPage = 1;

            loadVocabulary();

        }
    );


    // Previous page
    vocabularyPrev.addEventListener(
        "click",
        () => {

            if (vocabularyPage > 1) {

                vocabularyPage--;

                loadVocabulary();

            }

        }
    );


    // Next page
    vocabularyNext.addEventListener(
        "click",
        () => {

            if (vocabularyPage < vocabularyPages) {

                vocabularyPage++;

                loadVocabulary();

            }

        }
    );


    // Initial vocabulary loading
    loadVocabulary();

}