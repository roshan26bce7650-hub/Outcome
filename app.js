(() => {
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const gameGrid = $("#gameGrid");
  const gameCards = $$(".game-card");
  const search = $("#gameSearch");
  const filters = $$(".filter-pill");
  const emptyGames = $("#emptyGames");
  let activeFilter = "All";
  let toastTimer;

  function showToast(message) {
    const toast = $("#toast");
    toast.textContent = message;
    toast.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove("show"), 2800);
  }

  function filterGames() {
    const term = search.value.trim().toLowerCase();
    let visible = 0;
    gameCards.forEach(card => {
      const matchesTerm = `${card.dataset.game} ${card.dataset.category} ${card.dataset.description}`.toLowerCase().includes(term);
      const matchesFilter = activeFilter === "All" || card.dataset.category === activeFilter;
      const show = matchesTerm && matchesFilter;
      card.hidden = !show;
      if (show) visible++;
    });
    $("#catalogCount").textContent = `${visible} game${visible === 1 ? "" : "s"}`;
    emptyGames.hidden = visible !== 0;
  }
  search.addEventListener("input", filterGames);
  filters.forEach(button => button.addEventListener("click", () => {
    activeFilter = button.dataset.filter;
    filters.forEach(item => item.classList.toggle("selected", item === button));
    filterGames();
  }));
  $("#clearSearch").addEventListener("click", () => {
    search.value = ""; activeFilter = "All";
    filters.forEach(item => item.classList.toggle("selected", item.dataset.filter === "All"));
    filterGames(); search.focus();
  });

  function selectGame(name) {
    $("#gameSelect").value = name;
    gameCards.forEach(card => card.classList.toggle("selected-game", card.dataset.game === name));
    $("#predictor").scrollIntoView({behavior: "smooth", block: "start"});
    showToast(`${name} selected. Configure the matchup below.`);
  }
  gameCards.forEach(card => card.addEventListener("click", () => selectGame(card.dataset.game)));
  $("#gameSelect").addEventListener("change", () => {
    gameCards.forEach(card => card.classList.toggle("selected-game", card.dataset.game === $("#gameSelect").value));
  });

  const mobileMenu = $("#mobileMenu");
  mobileMenu.addEventListener("click", () => {
    const nav = $(".nav-links");
    const open = nav.classList.toggle("open");
    mobileMenu.setAttribute("aria-expanded", String(open));
    mobileMenu.textContent = open ? "×" : "☰";
  });
  $$(".nav-link").forEach(link => link.addEventListener("click", () => {
    $(".nav-links").classList.remove("open");
    mobileMenu.setAttribute("aria-expanded", "false");
    mobileMenu.textContent = "☰";
  }));
  $("#themeHint").addEventListener("click", () => showToast("Outcome is open-access. Real predictions become available after a relevant historical dataset is trained."));

  function linkRange(numberId, rangeId) {
    const number = $("#" + numberId), range = $("#" + rangeId);
    number.addEventListener("input", () => {
      if (number.value !== "") range.value = Math.max(Number(range.min), Math.min(Number(range.max), Number(number.value)));
    });
    range.addEventListener("input", () => { number.value = range.value; });
  }
  [["aWin","aWinRange"],["bWin","bWinRange"],["aForm","aFormRange"],["bForm","bFormRange"]].forEach(pair => linkRange(...pair));

  function setMessage(el, text, type = "info") {
    el.textContent = text; el.className = `form-message ${type}`; el.hidden = false;
  }
  function hideMessage(el) { el.hidden = true; el.textContent = ""; }

  async function refreshStatus() {
    try {
      const response = await fetch("/api/status");
      const data = await response.json();
      const trained = data.trained;
      $("#modelStatus").innerHTML = `<span class="status-dot ${trained ? "" : "status-muted"}"></span> ${trained ? "Model ready" : "Model not trained"}`;
      $("#statusPill").textContent = trained ? "MODEL READY" : "NOT TRAINED";
      $("#statusPill").classList.toggle("ready", trained);
      $("#modelSummary").textContent = trained
        ? "A model has been trained from the uploaded historical dataset. Check whether the dataset matches your chosen game."
        : "No model is available yet. Upload relevant historical results to train one.";
      $("#trainedRows").textContent = trained ? Number(data.meta.rows).toLocaleString() : "—";
      $("#trainedAccuracy").textContent = trained ? `${(Number(data.meta.accuracy) * 100).toFixed(1)}%` : "—";
      $("#trainedAt").textContent = trained ? new Date(data.meta.trained_at).toLocaleString() : "—";
      return data;
    } catch (error) {
      $("#modelSummary").textContent = "Could not connect to the model service. Check that the server is running.";
      return null;
    }
  }
  $("#refreshStatus").addEventListener("click", async () => {
    await refreshStatus(); showToast("Model status refreshed.");
  });

  $("#predictForm").addEventListener("submit", async event => {
    event.preventDefault();
    const message = $("#predictMessage");
    hideMessage(message);
    const a = $("#participantA").value.trim(), b = $("#participantB").value.trim();
    if (!a || !b || a.toLowerCase() === b.toLowerCase()) {
      setMessage(message, "Enter two different competitor or team names.", "error"); return;
    }
    const payload = {
      game: $("#gameSelect").value, participant_a: a, participant_b: b,
      a_win_rate: Number($("#aWin").value) / 100, b_win_rate: Number($("#bWin").value) / 100,
      a_recent_form: Number($("#aForm").value) / 100, b_recent_form: Number($("#bForm").value) / 100,
      a_rank: Number($("#aRank").value), b_rank: Number($("#bRank").value),
      a_rest_days: Number($("#aRest").value), b_rest_days: Number($("#bRest").value),
      a_home_advantage: $("#homeAdvantage").checked ? 1 : 0
    };
    if ([payload.a_win_rate,payload.b_win_rate,payload.a_recent_form,payload.b_recent_form].some(n => !Number.isFinite(n) || n < 0 || n > 1) ||
        [payload.a_rank,payload.b_rank,payload.a_rest_days,payload.b_rest_days].some(n => !Number.isFinite(n) || n < 0)) {
      setMessage(message, "Check the statistics. Rates must be 0–100%, and ranks/rest days cannot be negative.", "error"); return;
    }
    const button = $("#predictButton");
    button.disabled = true; button.textContent = "Evaluating matchup…";
    try {
      const response = await fetch("/api/predict", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(payload)});
      const data = await response.json();
      if (!response.ok) {
        setMessage(message, data.error || "Prediction could not be generated.", response.status === 409 ? "info" : "error");
        if (data.needs_training) $("#model-lab").scrollIntoView({behavior:"smooth",block:"start"});
        return;
      }
      renderResults(data);
      $("#results").hidden = false;
      $("#results").scrollIntoView({behavior:"smooth",block:"start"});
    } catch (error) {
      setMessage(message, "Unable to reach the prediction service. Please try again.", "error");
    } finally {
      button.disabled = false;
      button.innerHTML = 'Generate prediction <span aria-hidden="true">↗</span>';
    }
  });

  function renderResults(data) {
    $("#resultSubtitle").textContent = `${data.game} · ${data.participant_a} vs ${data.participant_b}`;
    $("#winnerName").textContent = data.top_outcome;
    $("#winnerProbability").innerHTML = `${Number(data.top_probability).toFixed(1)}<small>%</small>`;
    $("#winnerBar").style.width = `${Math.max(0,Math.min(100,Number(data.top_probability)))}%`;
    $("#winnerNote").textContent = "Highest probability in this model output. It does not mean the outcome is certain.";
    $("#probabilityBars").innerHTML = "";
    data.probabilities.forEach(item => {
      const row = document.createElement("div");
      row.className = "outcome-row";
      const head = document.createElement("div"); head.className = "outcome-bar-head";
      const label = document.createElement("span"); label.textContent = item.label;
      const percent = document.createElement("strong"); percent.textContent = `${Number(item.probability).toFixed(1)}%`;
      head.append(label, percent);
      const track = document.createElement("div"); track.className = "outcome-track";
      const fill = document.createElement("span"); fill.style.width = `${Math.max(0,Math.min(100,Number(item.probability)))}%`;
      track.append(fill); row.append(head,track); $("#probabilityBars").append(row);
    });
    $("#modelAccuracy").textContent = data.model_accuracy == null ? "N/A" : `${(Number(data.model_accuracy)*100).toFixed(1)}%`;
    $("#latency").textContent = `${data.latency_ms} ms`;
    $("#resultDisclaimer").textContent = data.disclaimer || "Model estimates are not guarantees.";
    hideMessage($("#predictMessage"));
  }
  $("#resetPrediction").addEventListener("click", () => {
    $("#results").hidden = true;
    $("#predictForm").scrollIntoView({behavior:"smooth",block:"start"});
    $("#participantA").focus({preventScroll:true});
  });

  const fileInput = $("#datasetFile"), dropzone = $("#dropzone");
  fileInput.addEventListener("change", () => {
    $("#fileName").textContent = fileInput.files[0] ? fileInput.files[0].name : "Choose a CSV file";
  });
  ["dragenter","dragover"].forEach(type => dropzone.addEventListener(type, event => {
    event.preventDefault(); dropzone.classList.add("dragover");
  }));
  ["dragleave","drop"].forEach(type => dropzone.addEventListener(type, event => {
    event.preventDefault(); dropzone.classList.remove("dragover");
  }));
  dropzone.addEventListener("drop", event => {
    const file = event.dataTransfer.files[0];
    if (file && file.name.toLowerCase().endsWith(".csv")) {
      try { const dt = new DataTransfer(); dt.items.add(file); fileInput.files = dt.files; $("#fileName").textContent = file.name; }
      catch (error) { showToast("Use Browse files to select the CSV on this browser."); }
    } else showToast("Please drop a .csv file.");
  });

  $("#uploadForm").addEventListener("submit", async event => {
    event.preventDefault();
    const message = $("#trainMessage"); hideMessage(message);
    if (!fileInput.files.length) { setMessage(message, "Choose a CSV file first.", "error"); return; }
    if (!fileInput.files[0].name.toLowerCase().endsWith(".csv")) { setMessage(message, "Please select a CSV file.", "error"); return; }
    const button = $("#trainButton"); button.disabled = true;
    button.textContent = "Training model…"; $("#trainingProgress").hidden = false;
    const formData = new FormData(); formData.append("dataset", fileInput.files[0]);
    try {
      const response = await fetch("/api/train", {method:"POST", body:formData});
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Training failed.");
      setMessage(message, `Training complete. ${data.meta.rows} rows · ${(data.meta.accuracy*100).toFixed(1)}% held-out accuracy. This score is only an estimate of future performance.`, "success");
      await refreshStatus();
      showToast("Model trained successfully.");
    } catch (error) {
      setMessage(message, error.message || "Training failed.", "error");
    } finally {
      button.disabled = false; button.innerHTML = 'Train model <span aria-hidden="true">↗</span>';
      $("#trainingProgress").hidden = true;
    }
  });

  // Smooth anchor behavior for in-page navigation and initial model state.
  $$('a[href^="#"]').forEach(link => link.addEventListener("click", event => {
    const target = $(link.getAttribute("href"));
    if (target) { event.preventDefault(); target.scrollIntoView({behavior:"smooth",block:"start"}); history.replaceState(null,"",link.getAttribute("href")); }
  }));
  refreshStatus();
})();
