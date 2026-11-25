/** My Health: the patient's prescriptions and search history. */
import { clearHistory, deleteDocument, deleteEverything, deleteSearch, fetchDocuments, fetchHistory, openAnswer, uploadDocument } from "./api.js";

const $ = (id) => document.getElementById(id);

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text; // textContent: names and questions are user input
  return node;
}

function say(text, isError = false) {
  $("pageMessage").textContent = text;
  $("pageMessage").classList.toggle("error", isError);
}

const when = (seconds) => new Date(seconds * 1000).toLocaleString([], { dateStyle: "medium", timeStyle: "short" });

async function run(action, done) {
  try {
    await action();
    if (done) say(done);
  } catch (err) {
    say(err.message, true);
  }
}

async function showDocuments() {
  const docs = await fetchDocuments();
  const list = $("documentList");
  list.replaceChildren();
  if (!docs.length) list.append(el("p", "text-subtitle", "No documents yet."));
  docs.forEach((doc) => {
    const card = el("article", "private-card");
    const row = el("div", "private-row");
    row.append(el("strong", "", doc.filename));
    const remove = el("button", "btn-pill btn-outline btn-small", "Delete");
    remove.type = "button";
    remove.setAttribute("aria-label", `Delete ${doc.filename}`);
    remove.addEventListener("click", () => run(async () => {
      await deleteDocument(doc.id);
      await showDocuments();
    }, `${doc.filename} deleted.`));
    row.append(remove);
    card.append(row, el("p", "private-meta", `${doc.kind === "report" ? "Report" : "Prescription"}, ${doc.pages} page${doc.pages === 1 ? "" : "s"}, added ${when(doc.created)}`));
    const found = [];
    if (doc.medicines.length) found.push(`Medicines found: ${doc.medicines.join(", ")}`);
    if (doc.tests.length) found.push(`Test results found: ${doc.tests.join(", ")}`);
    if (!found.length) found.push("No medicines or test values recognised. You can still ask about this document.");
    found.forEach((text) => card.append(el("p", "", text)));
    const details = el("details", "");
    details.append(el("summary", "", "Show the text HealthNest read"), el("pre", "private-text", doc.text));
    card.append(details);
    list.append(card);
  });
}

async function showHistory() {
  const items = await fetchHistory();
  const list = $("historyList");
  list.replaceChildren();
  $("btnClearHistory").hidden = !items.length;
  if (!items.length) list.append(el("li", "text-subtitle", "No searches yet."));
  items.forEach((item) => {
    const row = el("li", "private-row");
    const ask = el("button", "link-btn", item.query);
    ask.type = "button";
    ask.addEventListener("click", () => openAnswer(item.query));
    const remove = el("button", "btn-pill btn-outline btn-small", "Remove");
    remove.type = "button";
    remove.setAttribute("aria-label", `Remove search: ${item.query}`);
    remove.addEventListener("click", () => run(async () => {
      await deleteSearch(item.id);
      await showHistory();
    }));
    const left = el("div", "private-grow");
    left.append(ask, el("p", "private-meta", when(item.created)));
    row.append(left, remove);
    list.append(row);
  });
}

$("pdfInput").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  say(`Reading ${file.name}…`);
  await run(async () => {
    const doc = await uploadDocument(file);
    await showDocuments();
    say(`${doc.filename} added. Please check that the text was read correctly.`);
  });
  e.target.value = "";
});

$("btnClearHistory").addEventListener("click", () => {
  if (!window.confirm("Clear your whole search history? This cannot be undone.")) return;
  run(async () => { await clearHistory(); await showHistory(); }, "Search history cleared.");
});

$("btnDeleteAll").addEventListener("click", () => {
  if (!window.confirm("Delete all your documents and search history? This cannot be undone.")) return;
  run(async () => {
    await deleteEverything();
    sessionStorage.removeItem("hn_question");
    await Promise.all([showDocuments(), showHistory()]);
  }, "All your data was deleted.");
});

run(() => Promise.all([showDocuments(), showHistory()]));
