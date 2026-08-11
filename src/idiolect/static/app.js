const token = document.querySelector('meta[name="idiolect-token"]').content;
const apiHeaders = { "X-Idiolect-Token": token };

const elements = {
  status: document.getElementById("codex-status"),
  error: document.getElementById("error-box"),
  form: document.getElementById("corpus-form"),
  files: document.getElementById("files"),
  pasted: document.getElementById("pasted-text"),
  pastedAuthorship: document.getElementById("pasted-authorship-confirmation"),
  pasteCount: document.getElementById("paste-count"),
  sampleList: document.getElementById("sample-list"),
  generate: document.getElementById("generate-button"),
  progressPanel: document.getElementById("progress-panel"),
  progressStage: document.getElementById("progress-stage"),
  progressMessage: document.getElementById("progress-message"),
  cancel: document.getElementById("cancel-button"),
  reviewPanel: document.getElementById("review-panel"),
  profileSummary: document.getElementById("profile-summary"),
  corpusSummary: document.getElementById("corpus-summary"),
  warnings: document.getElementById("warning-list"),
  traits: document.getElementById("trait-list"),
  build: document.getElementById("build-button"),
  resultPanel: document.getElementById("result-panel"),
  download: document.getElementById("download-button"),
  install: document.getElementById("install-button"),
  installResult: document.getElementById("install-result"),
  preview: document.getElementById("skill-preview"),
};

let codexReady = false;
let samples = [];
let currentJob = null;
let polling = false;
let pollFailures = 0;

function rememberJob(jobId) {
  currentJob = jobId;
  if (jobId) {
    window.sessionStorage.setItem("idiolect-job-id", jobId);
  } else {
    window.sessionStorage.removeItem("idiolect-job-id");
  }
}

function setError(message) {
  if (!message) {
    elements.error.hidden = true;
    elements.error.textContent = "";
    return;
  }
  elements.error.textContent = message;
  elements.error.hidden = false;
  elements.error.focus();
}

async function responseJson(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload.detail || "The local app could not complete that request.");
    error.status = response.status;
    throw error;
  }
  return payload;
}

async function checkCodex() {
  try {
    const response = await fetch("/api/status", { headers: apiHeaders });
    const result = await responseJson(response);
    codexReady = Boolean(result.ready);
    elements.status.classList.toggle("ready", codexReady);
    elements.status.classList.toggle("not-ready", !codexReady);
    elements.status.lastChild.textContent = ` ${result.message}`;
    elements.status.title = result.version || "";
  } catch (error) {
    codexReady = false;
    elements.status.classList.add("not-ready");
    elements.status.lastChild.textContent = " Codex check failed";
  }
  elements.generate.disabled = !codexReady || Boolean(currentJob);
}

function option(value, label, selected) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = label;
  item.selected = selected;
  return item;
}

function compactSelect(labelText, choices, value, onChange) {
  const wrapper = document.createElement("div");
  wrapper.className = "compact-field";
  const label = document.createElement("label");
  label.textContent = labelText;
  const select = document.createElement("select");
  select.id = `sample-field-${crypto.randomUUID()}`;
  label.htmlFor = select.id;
  choices.forEach(([choiceValue, choiceLabel]) => {
    select.append(option(choiceValue, choiceLabel, choiceValue === value));
  });
  select.addEventListener("change", () => onChange(select.value));
  wrapper.append(label, select);
  return wrapper;
}

function renderSamples() {
  elements.sampleList.replaceChildren();
  samples.forEach((sample, index) => {
    const card = document.createElement("div");
    card.className = "sample-card";

    const name = document.createElement("div");
    name.className = "sample-name";
    const strong = document.createElement("strong");
    strong.textContent = sample.file.name;
    strong.title = sample.file.name;
    const size = document.createElement("small");
    size.textContent = `${Math.max(1, Math.round(sample.file.size / 1024))} KB`;
    name.append(strong, size);

    const authorship = compactSelect(
      "Authorship",
      [
        ["self", "Fully mine"],
        ["coauthored", "Coauthored"],
        ["edited", "Heavily edited"],
        ["unknown", "Unknown"],
      ],
      sample.authorship,
      (value) => { sample.authorship = value; },
    );
    const genre = compactSelect(
      "Text type",
      [
        ["auto", "Infer"],
        ["email", "Email"],
        ["article", "Article"],
        ["social-post", "Social post"],
        ["technical-note", "Technical note"],
        ["academic", "Academic"],
        ["other", "Other"],
      ],
      sample.genre,
      (value) => { sample.genre = value; },
    );
    const language = compactSelect(
      "Language",
      [
        ["auto", "Infer"],
        ["en", "English"],
        ["de-DE", "German (DE)"],
        ["de-AT", "German (AT)"],
        ["de-CH", "German (CH)"],
        ["other", "Other"],
      ],
      sample.language,
      (value) => { sample.language = value; },
    );
    const era = compactSelect(
      "Voice era",
      [
        ["current", "Current"],
        ["archive", "Older / exclude"],
      ],
      sample.current_voice ? "current" : "archive",
      (value) => { sample.current_voice = value === "current"; },
    );

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "remove-button";
    remove.textContent = "×";
    remove.setAttribute("aria-label", `Remove ${sample.file.name}`);
    remove.addEventListener("click", () => {
      samples.splice(index, 1);
      renderSamples();
    });
    card.append(name, authorship, genre, language, era, remove);
    elements.sampleList.append(card);
  });
}

elements.files.addEventListener("change", () => {
  const incoming = Array.from(elements.files.files || []);
  incoming.forEach((file) => {
    const duplicate = samples.some((sample) => sample.file.name === file.name && sample.file.size === file.size);
    if (!duplicate) {
      samples.push({ file, authorship: "self", genre: "auto", language: "auto", current_voice: true });
    }
  });
  elements.files.value = "";
  renderSamples();
});

elements.pasted.addEventListener("input", () => {
  const count = (elements.pasted.value.match(/\S+/g) || []).length;
  elements.pasteCount.textContent = `${count.toLocaleString()} estimated words`;
  const hasPastedText = Boolean(elements.pasted.value.trim());
  elements.pastedAuthorship.disabled = !hasPastedText;
  if (!hasPastedText) elements.pastedAuthorship.checked = false;
});

function showProgress(job) {
  elements.progressPanel.hidden = false;
  elements.progressStage.textContent = job.stage.replaceAll("_", " ");
  elements.progressMessage.textContent = job.message;
  elements.cancel.disabled = !["queued", "analyzing", "validating"].includes(job.stage);
  document.querySelectorAll(".stage-list li").forEach((item) => {
    item.classList.toggle("active", item.dataset.stage === job.stage);
  });
}

function renderWarnings(stats, uncertainties) {
  elements.warnings.replaceChildren();
  [...(stats.warnings || []), ...(uncertainties || [])].forEach((warning) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = warning;
    elements.warnings.append(paragraph);
  });
}

function renderReview(job) {
  elements.progressPanel.hidden = true;
  elements.reviewPanel.hidden = false;
  elements.profileSummary.textContent = job.profile.summary;
  const stats = job.stats;
  elements.corpusSummary.textContent = `${stats.clean_words.toLocaleString()} clean words · ${stats.included_documents} documents · ${stats.document_families} families · ${stats.quality} profile`;
  renderWarnings(stats, job.profile.uncertainties);
  elements.traits.replaceChildren();
  job.profile.traits.forEach((trait) => {
    const label = document.createElement("label");
    label.className = "trait-card";
    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = trait.id;
    checkbox.checked = false;
    checkbox.addEventListener("change", updateBuildState);
    const body = document.createElement("div");
    const meta = document.createElement("div");
    meta.className = "trait-meta";
    [trait.mode, trait.dimension, trait.strength, `${Math.round(trait.confidence * 100)}%`, `${trait.evidence_count} excerpts`].forEach((value) => {
      const span = document.createElement("span");
      span.textContent = value;
      meta.append(span);
    });
    const text = document.createElement("p");
    text.textContent = trait.instruction;
    const scope = document.createElement("p");
    scope.className = "trait-scope";
    scope.textContent = `Applies to languages: ${trait.scope.languages.join(", ")}; text types: ${trait.scope.genres.join(", ")}; registers: ${trait.scope.registers.join(", ")}.`;
    body.append(meta, text, scope);
    label.append(checkbox, body);
    elements.traits.append(label);
  });
  updateBuildState();
  const heading = document.getElementById("review-title");
  heading.focus({ preventScroll: true });
  heading.scrollIntoView({ block: "start" });
}

function updateBuildState() {
  const selected = elements.traits.querySelectorAll('input[type="checkbox"]:checked').length;
  elements.build.disabled = selected < 3;
}

function renderResult(job) {
  elements.progressPanel.hidden = true;
  elements.reviewPanel.hidden = true;
  elements.resultPanel.hidden = false;
  elements.preview.textContent = job.preview;
  elements.generate.disabled = !codexReady;
  const heading = document.getElementById("result-title");
  heading.focus({ preventScroll: true });
  heading.scrollIntoView({ block: "start" });
}

async function pollJob() {
  if (!currentJob || polling) return;
  polling = true;
  try {
    const response = await fetch(`/api/jobs/${currentJob}`, { headers: apiHeaders });
    const job = await responseJson(response);
    pollFailures = 0;
    showProgress(job);
    if (job.stage === "review") {
      renderReview(job);
      return;
    }
    if (job.stage === "complete") {
      renderResult(job);
      return;
    }
    if (job.stage === "failed" || job.stage === "cancelled") {
      rememberJob(null);
      setError(job.error || job.message);
      elements.generate.disabled = !codexReady;
      return;
    }
    window.setTimeout(pollJob, 1200);
  } catch (error) {
    if ([400, 404].includes(error.status)) {
      rememberJob(null);
      setError(error.message);
      elements.generate.disabled = !codexReady;
      return;
    }
    pollFailures += 1;
    const delay = Math.min(15000, 1000 * (2 ** Math.min(pollFailures, 4)));
    elements.progressMessage.textContent = "Connection interrupted. Retrying while the local Codex job continues.";
    window.setTimeout(pollJob, delay);
  } finally {
    polling = false;
  }
}

elements.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setError("");
  if (!codexReady) {
    setError("Codex is not signed in with ChatGPT. Run `codex login` in a terminal.");
    return;
  }
  if (!elements.form.reportValidity()) return;
  if (!samples.length && !elements.pasted.value.trim()) {
    setError("Add at least one text file or pasted sample.");
    return;
  }
  if (elements.pasted.value.trim() && !elements.pastedAuthorship.checked) {
    setError("Confirm that the pasted text is fully self-authored and represents your current voice.");
    return;
  }
  const form = new FormData();
  form.append("skill_name", document.getElementById("skill-name").value);
  form.append("voice_label", document.getElementById("voice-label").value);
  form.append("purpose", document.getElementById("purpose").value);
  form.append("pasted_text", elements.pasted.value);
  form.append("pasted_self_authored", String(elements.pastedAuthorship.checked));
  form.append("rights_confirmed", "true");
  form.append("sample_metadata", JSON.stringify(samples.map(({ authorship, genre, language, current_voice }) => ({ authorship, genre, language, current_voice }))));
  samples.forEach((sample) => form.append("files", sample.file, sample.file.name));
  elements.generate.disabled = true;
  elements.resultPanel.hidden = true;
  elements.reviewPanel.hidden = true;
  elements.progressPanel.hidden = false;
  try {
    const response = await fetch("/api/jobs", {
      method: "POST",
      headers: { ...apiHeaders, "Idempotency-Key": crypto.randomUUID() },
      body: form,
    });
    const job = await responseJson(response);
    rememberJob(job.id);
    showProgress(job);
    window.setTimeout(pollJob, 500);
  } catch (error) {
    setError(error.message);
    elements.generate.disabled = !codexReady;
    elements.progressPanel.hidden = true;
  }
});

elements.cancel.addEventListener("click", async () => {
  if (!currentJob) return;
  elements.cancel.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${currentJob}`, {
      method: "DELETE",
      headers: apiHeaders,
    });
    const job = await responseJson(response);
    rememberJob(null);
    showProgress(job);
    setError(job.message);
    elements.generate.disabled = !codexReady;
  } catch (error) {
    setError(error.message);
  }
});

elements.build.addEventListener("click", async () => {
  if (!currentJob) return;
  setError("");
  const accepted = Array.from(elements.traits.querySelectorAll('input[type="checkbox"]:checked')).map((item) => item.value);
  if (accepted.length < 3) {
    setError("Keep at least three evidenced traits before building the skill.");
    return;
  }
  elements.build.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${currentJob}/compile`, {
      method: "POST",
      headers: { ...apiHeaders, "Content-Type": "application/json" },
      body: JSON.stringify({ accepted_trait_ids: accepted }),
    });
    const job = await responseJson(response);
    renderResult(job);
  } catch (error) {
    setError(error.message);
    updateBuildState();
  }
});

elements.download.addEventListener("click", async () => {
  if (!currentJob) return;
  try {
    const response = await fetch(`/api/jobs/${currentJob}/download`, { headers: apiHeaders });
    if (!response.ok) await responseJson(response);
    const blob = await response.blob();
    const disposition = response.headers.get("Content-Disposition") || "";
    const match = disposition.match(/filename="([^"]+)"/);
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = match ? match[1] : "idiolect-skill.zip";
    document.body.append(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(link.href);
  } catch (error) {
    setError(error.message);
  }
});

elements.install.addEventListener("click", async () => {
  if (!currentJob) return;
  const confirmed = window.confirm("Install this skill into your personal ~/.agents/skills directory? Existing skills are never overwritten.");
  if (!confirmed) return;
  elements.install.disabled = true;
  try {
    const response = await fetch(`/api/jobs/${currentJob}/install`, {
      method: "POST",
      headers: apiHeaders,
    });
    const result = await responseJson(response);
    elements.installResult.textContent = `${result.message} ${result.path}`;
  } catch (error) {
    setError(error.message);
    elements.install.disabled = false;
  }
});

async function initialize() {
  rememberJob(window.sessionStorage.getItem("idiolect-job-id"));
  await checkCodex();
  if (currentJob) {
    elements.progressPanel.hidden = false;
    elements.generate.disabled = true;
    pollJob();
  }
}

initialize();
