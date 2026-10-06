import { fetchEvalSummary } from "./api.js";

document.addEventListener("DOMContentLoaded", async () => {
  const evalContainer = document.getElementById("evalContainer");

  try {
    const data = await fetchEvalSummary();
    if (data.status === "pending" || !data.retrieval_eval) {
      evalContainer.innerHTML = `
        <div style="background:#edf2f7; padding:20px; border-radius:12px;">
          <p>Evaluation metrics benchmark is currently running. Run <code>python eval/run_retrieval_eval.py</code> to generate the latest report card.</p>
        </div>
      `;
      return;
    }

    // Render report card tables
    let html = `
      <table style="width:100%; border-collapse:collapse; margin-bottom:24px;">
        <thead>
          <tr style="background:#f7fafc; border-bottom:2px solid #e2e8f0; text-align:left;">
            <th style="padding:10px;">Retriever</th>
            <th style="padding:10px;">P@1</th>
            <th style="padding:10px;">P@5</th>
            <th style="padding:10px;">Recall@5</th>
            <th style="padding:10px;">MRR</th>
          </tr>
        </thead>
        <tbody>
    `;

    Object.entries(data.retrieval_eval).forEach(([model, metrics]) => {
      html += `
        <tr style="border-bottom:1px solid #edf2f7;">
          <td style="padding:10px;"><strong>${model.toUpperCase()}</strong></td>
          <td style="padding:10px;">${(metrics["P@1"] * 100).toFixed(1)}%</td>
          <td style="padding:10px;">${(metrics["P@5"] * 100).toFixed(1)}%</td>
          <td style="padding:10px;">${(metrics["Recall@5"] * 100).toFixed(1)}%</td>
          <td style="padding:10px;">${metrics["MRR"].toFixed(3)}</td>
        </tr>
      `;
    });

    html += `</tbody></table>`;
    evalContainer.innerHTML = html;

  } catch (err) {
    evalContainer.innerHTML = `<p style="color:red;">Error loading eval summary: ${err.message}</p>`;
  }
});
