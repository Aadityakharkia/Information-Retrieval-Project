// Frontend Logic for Trustworthy Medical RAG & IR Diagnostics

// Default thresholds per retriever
const DEFAULT_THRESHOLDS = {
  inverted_index: 0.28,
  bm25: 8.0,
  dense: 0.35,
  hybrid: 0.015
};

const THRESHOLD_RANGES = {
  inverted_index: { min: 0.0, max: 1.0, step: 0.01, default: 0.28 },
  bm25: { min: 0.0, max: 30.0, step: 0.5, default: 8.0 },
  dense: { min: 0.0, max: 1.0, step: 0.01, default: 0.35 },
  hybrid: { min: 0.0, max: 0.1, step: 0.001, default: 0.015 }
};

// Event listener for retriever change
document.getElementById('retrieverSelect').addEventListener('change', function(e) {
  const ret = e.target.value;
  const cfg = THRESHOLD_RANGES[ret] || THRESHOLD_RANGES.inverted_index;
  const slider = document.getElementById('thresholdSlider');
  slider.min = cfg.min;
  slider.max = cfg.max;
  slider.step = cfg.step;
  slider.value = cfg.default;
  document.getElementById('thresholdValueText').textContent = cfg.default;
});

function setQuery(text) {
  document.getElementById('queryInput').value = text;
}

function updateThresholdValue(val) {
  document.getElementById('thresholdValueText').textContent = val;
}

function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  
  const activeBtn = Array.from(document.querySelectorAll('.tab-btn')).find(b => 
    b.getAttribute('onclick') && b.getAttribute('onclick').includes(tabId)
  );
  if (activeBtn) activeBtn.classList.add('active');
  
  const target = document.getElementById(tabId);
  if (target) target.classList.add('active');
}

async function submitQuery() {
  const query = document.getElementById('queryInput').value.trim();
  if (!query) {
    alert("Please enter a medical question.");
    return;
  }

  const retriever = document.getElementById('retrieverSelect').value;
  const model = document.getElementById('generatorSelect').value;
  const threshold = parseFloat(document.getElementById('thresholdSlider').value);
  const topK = parseInt(document.getElementById('topKSelect').value);

  // UI loading state
  const btn = document.getElementById('submitQueryBtn');
  const btnText = document.getElementById('btnText');
  const spinner = document.getElementById('btnSpinner');
  btn.disabled = true;
  btnText.textContent = "Retrieving & Auditing...";
  spinner.classList.remove('hidden');

  try {
    const res = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: query,
        retriever_type: retriever,
        top_k: topK,
        threshold: threshold,
        generator_model: model
      })
    });

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Query failed");
    }

    const data = await res.json();
    renderResults(data);

  } catch (err) {
    alert("Error executing query: " + err.message);
  } finally {
    btn.disabled = false;
    btnText.textContent = "Execute RAG & IR Inspection";
    spinner.classList.add('hidden');
  }
}

function renderResults(data) {
  // 1. Answer Card & Refusal Banner
  const answerTextEl = document.getElementById('answerText');
  const refusalBanner = document.getElementById('refusalBanner');
  const refusalReasonText = document.getElementById('refusalReasonText');
  const statusBadge = document.getElementById('answerStatusBadge');
  const metaBar = document.getElementById('answerMetaBar');

  metaBar.style.display = 'flex';
  document.getElementById('metaRetriever').textContent = data.retriever_type;
  document.getElementById('metaScore').textContent = data.top_retrieval_score.toFixed(4);
  document.getElementById('metaThreshold').textContent = data.threshold_used.toFixed(4);
  document.getElementById('metaModel').textContent = data.generator_model;

  if (data.refused) {
    refusalBanner.classList.remove('hidden');
    refusalReasonText.textContent = data.refusal_reason || "Score below threshold.";
    statusBadge.textContent = "Refused (Safe)";
    statusBadge.className = "badge badge-warning";
    answerTextEl.innerHTML = `<p style="color: #fde68a;">${data.answer}</p>`;
  } else {
    refusalBanner.classList.add('hidden');
    statusBadge.textContent = "Answer Generated";
    statusBadge.className = "badge badge-success";

    // Format inline citations with highlighted interactive pills
    const highlighted = data.answer.replace(/\[C(\d+)\]/g, '<span class="citation-tag" onclick="highlightChunk($1)">[C$1]</span>');
    answerTextEl.innerHTML = highlighted;
  }

  // 2. Citation Audit Card
  const auditCard = document.getElementById('auditCard');
  if (!data.refused && data.citation_audit && data.citation_audit.sentence_audits.length > 0) {
    auditCard.style.display = 'block';
    const audit = data.citation_audit;

    document.getElementById('faithfulnessScoreText').textContent = `${(audit.faithfulness_score * 100).toFixed(1)}%`;
    document.getElementById('countSupported').textContent = audit.supported_count;
    document.getElementById('countPartial').textContent = audit.partially_supported_count;
    document.getElementById('countUnsupported').textContent = audit.unsupported_count;
    document.getElementById('countUncited').textContent = audit.uncited_count;

    const listEl = document.getElementById('sentenceAuditList');
    listEl.innerHTML = '';

    audit.sentence_audits.forEach(s => {
      const row = document.createElement('div');
      row.className = `sentence-audit-row verdict-${s.verdict}`;

      const badgeColor = s.verdict === 'SUPPORTED' ? 'badge-success' :
                         s.verdict === 'PARTIALLY_SUPPORTED' ? 'badge-warning' :
                         s.verdict === 'UNSUPPORTED' ? 'badge-danger' : 'badge-neutral';

      row.innerHTML = `
        <div class="sentence-header">
          <span class="badge ${badgeColor}">${s.verdict}</span>
          <div class="sentence-scores">
            <span>Score: <strong>${s.confidence_score.toFixed(3)}</strong></span>
            <span>Cosine: <strong>${s.cosine_similarity.toFixed(3)}</strong></span>
            <span>Cov: <strong>${(s.term_coverage * 100).toFixed(0)}%</strong></span>
          </div>
        </div>
        <div class="sentence-text">${s.sentence}</div>
        <div class="sentence-note">${s.explanation}</div>
      `;
      listEl.appendChild(row);
    });
  } else {
    auditCard.style.display = 'none';
  }

  // 3. Deep IR Diagnostics Section
  const irSection = document.getElementById('irSection');
  irSection.style.display = 'block';

  renderIRDiagnostics(data.ir_diagnostics, data.retrieved_chunks);
}

function renderIRDiagnostics(diag, chunks) {
  // Tab 1: Vector Space Diagnostics
  document.getElementById('diagTokens').textContent = (diag.tokens || []).join(', ') || '-';
  document.getElementById('diagStems').textContent = (diag.stems || []).join(', ') || '-';
  document.getElementById('diagNorm').textContent = diag.query_norm !== undefined ? diag.query_norm : '-';

  const tbody = document.getElementById('termWeightsBody');
  tbody.innerHTML = '';

  if (diag.term_diagnostics) {
    for (const [term, d] of Object.entries(diag.term_diagnostics)) {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong style="color: #22d3ee;">${term}</strong></td>
        <td>${d.df || 0}</td>
        <td>${d.idf !== undefined ? d.idf.toFixed(4) : '-'}</td>
        <td>${d.raw_tf || 1}</td>
        <td><strong>${d.normalized_ltc_weight !== undefined ? d.normalized_ltc_weight.toFixed(4) : '-'}</strong></td>
        <td>${d.postings_count || 0}</td>
        <td><button class="inspect-btn" onclick="inspectTermPostings('${term}')">View Postings</button></td>
      `;
      tbody.appendChild(tr);
    }
  }

  // Tab 2: Retrieved Chunks
  const chunksContainer = document.getElementById('chunksContainer');
  chunksContainer.innerHTML = '';

  chunks.forEach((c, idx) => {
    const card = document.createElement('div');
    card.className = 'chunk-card';
    card.id = `chunk-card-${idx + 1}`;
    card.innerHTML = `
      <div class="chunk-header">
        <div>
          <span class="chunk-tag">[C${idx + 1}]</span>
          <strong style="margin-left: 8px;">${c.chunk_id}</strong>
          <span style="color: #94a3b8; margin-left: 8px;">(doc_id: ${c.doc_id}, qtype: ${c.qtype})</span>
        </div>
        <div class="chunk-score">
          Score: <strong>${c.score.toFixed(4)}</strong> (Rank #${c.rank})
        </div>
      </div>
      <div class="chunk-body">${c.text}</div>
    `;
    chunksContainer.appendChild(card);
  });

  // Tab 3: Rank Fusion or BM25 Scoring
  const fusionContainer = document.getElementById('fusionDiagnosticsContainer');
  if (diag.top_fused_results) {
    let tableHtml = `
      <h3 style="color: #22d3ee; margin-bottom: 10px;">Reciprocal Rank Fusion (RRF: k=${diag.rrf_k_constant})</h3>
      <table class="ir-table">
        <thead>
          <tr>
            <th>Chunk ID</th>
            <th>Fused Rank</th>
            <th>Fused RRF Score</th>
            <th>Sparse Rank</th>
            <th>Sparse Contrib</th>
            <th>Dense Rank</th>
            <th>Dense Contrib</th>
          </tr>
        </thead>
        <tbody>
    `;
    diag.top_fused_results.forEach(r => {
      tableHtml += `
        <tr>
          <td><strong>${r.chunk_id}</strong></td>
          <td>#${r.fused_rank}</td>
          <td><strong>${r.fused_rrf_score.toFixed(6)}</strong></td>
          <td>${r.sparse_rank ? '#' + r.sparse_rank : 'None'}</td>
          <td>${r.sparse_rrf_contrib.toFixed(6)}</td>
          <td>${r.dense_rank ? '#' + r.dense_rank : 'None'}</td>
          <td>${r.dense_rrf_contrib.toFixed(6)}</td>
        </tr>
      `;
    });
    tableHtml += `</tbody></table>`;
    fusionContainer.innerHTML = tableHtml;
  } else if (diag.top_results_breakdown) {
    let bm25Html = `
      <h3 style="color: #22d3ee; margin-bottom: 10px;">Okapi BM25 Scoring Decomposition (k1=${diag.k1}, b=${diag.b}, avgdl=${diag.avg_doc_len})</h3>
      <div style="display: flex; flex-direction: column; gap: 10px;">
    `;
    diag.top_results_breakdown.forEach(r => {
      bm25Html += `
        <div class="chunk-card">
          <div class="chunk-header">
            <strong>${r.chunk_id} (Rank #${r.rank})</strong>
            <span>Total BM25: <strong>${r.total_score.toFixed(4)}</strong> | DocLen: ${r.doc_length} (LenNorm: ${r.length_normalization_factor})</span>
          </div>
          <div style="font-size: 0.8rem; color: #94a3b8;">
            Term Contributions: ${JSON.stringify(r.term_contributions)}
          </div>
        </div>
      `;
    });
    bm25Html += `</div>`;
    fusionContainer.innerHTML = bm25Html;
  } else {
    fusionContainer.innerHTML = `<p class="helper-text">Diagnostics displayed for ${diag.retriever}.</p>`;
  }
}

function highlightChunk(chunkNum) {
  switchTab('tab-chunks');
  const target = document.getElementById(`chunk-card-${chunkNum}`);
  if (target) {
    target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    target.style.outline = '2px solid #06b6d4';
    setTimeout(() => { target.style.outline = 'none'; }, 2000);
  }
}

async function inspectTermPostings(term) {
  try {
    const res = await fetch(`/api/explain_term?term=${encodeURIComponent(term)}`);
    const data = await res.json();

    document.getElementById('modalTermTitle').textContent = `Inverted Index Postings for: "${data.term}"`;
    const body = document.getElementById('modalPostingsBody');
    body.innerHTML = `
      <div style="margin-bottom: 12px; font-size: 0.9rem;">
        <span>Document Frequency (DF): <strong>${data.df}</strong></span> &bull; 
        <span>IDF: <strong>${data.idf.toFixed(4)}</strong></span> &bull; 
        <span>Total Postings: <strong>${data.total_postings}</strong></span>
      </div>
      <p style="font-size: 0.8rem; color: #94a3b8; margin-bottom: 8px;">First 10 Postings Entries (chunk_idx, raw_tf):</p>
      <pre style="background: #0b0f19; padding: 12px; border-radius: 6px; font-family: monospace; font-size: 0.85rem; max-height: 250px; overflow-y: auto;">${JSON.stringify(data.postings_snippet, null, 2)}</pre>
    `;

    document.getElementById('postingsModal').classList.remove('hidden');
  } catch (err) {
    alert("Could not load postings: " + err.message);
  }
}

function closePostingsModal() {
  document.getElementById('postingsModal').classList.add('hidden');
}

// Live IR Typeahead Auto-Suggest Listener
document.addEventListener('DOMContentLoaded', () => {
  const queryInput = document.getElementById('queryInput');
  const suggestDropdown = document.getElementById('suggestDropdown');
  let debounceTimer = null;

  if (queryInput && suggestDropdown) {
    queryInput.addEventListener('input', (e) => {
      clearTimeout(debounceTimer);
      const val = e.target.value.trim();
      if (val.length < 1) {
        suggestDropdown.classList.remove('active');
        suggestDropdown.innerHTML = '';
        return;
      }

      debounceTimer = setTimeout(async () => {
        try {
          const res = await fetch(`/api/suggest?q=${encodeURIComponent(val)}`);
          if (!res.ok) return;
          const data = await res.json();
          const suggestions = data.suggestions || [];
          if (suggestions.length === 0) {
            suggestDropdown.classList.remove('active');
            suggestDropdown.innerHTML = '';
            return;
          }

          suggestDropdown.innerHTML = '';
          suggestions.forEach((item) => {
            const div = document.createElement('div');
            div.className = 'suggest-item';
            div.innerHTML = `
              <svg class="suggest-item-icon" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
                <circle cx="11" cy="11" r="8"/>
                <line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
              <span>${item}</span>
            `;
            div.addEventListener('click', () => {
              queryInput.value = item;
              suggestDropdown.classList.remove('active');
              submitQuery();
            });
            suggestDropdown.appendChild(div);
          });

          suggestDropdown.classList.add('active');
        } catch (err) {
          console.error("Auto-suggest error:", err);
        }
      }, 100);
    });

    document.addEventListener('click', (e) => {
      if (!suggestDropdown.contains(e.target) && e.target !== queryInput) {
        suggestDropdown.classList.remove('active');
      }
    });
  }
});
