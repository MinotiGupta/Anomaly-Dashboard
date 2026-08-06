const API_BASE = 'http://localhost:5000/api';

export async function uploadCSV(file) {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || 'Upload failed');
  }

  return res.json();
}

export async function runModel(filePath, useFeedback = false) {
  const res = await fetch(`${API_BASE}/run`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ path: filePath, use_feedback: useFeedback }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || 'Model execution failed');
  }

  return res.json();
}

export async function submitFeedback(runId, dataType, fileName, feedbackEntries) {
  const res = await fetch(`${API_BASE}/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      run_id: runId,
      data_type: dataType,
      file_name: fileName,
      feedback_entries: feedbackEntries,
    }),
  });

  if (!res.ok) {
    const err = await res.json();
    throw new Error(err.error || 'Feedback submission failed');
  }

  return res.json();
}

export async function getFeedbackSummary() {
  const res = await fetch(`${API_BASE}/feedback/summary`);
  if (!res.ok) throw new Error('Failed to fetch feedback summary');
  return res.json();
}
