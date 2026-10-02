(function () {
  const STORAGE_KEY = "herdline_herd_size";
  const LOADING_STEPS = [
    "Detecting goats…",
    "Tracking movement…",
    "Counting gate crossings…",
    "Building your summary…",
  ];

  const form = document.getElementById("count-form");
  const fileInput = document.getElementById("video-input");
  const fileLabel = document.getElementById("file-label");
  const overlay = document.getElementById("loading-overlay");
  const submitBtn = document.getElementById("submit-btn");
  const herdInput = document.getElementById("herd_size");
  const lineMode = document.getElementById("line_mode");
  const directionSelect = document.getElementById("direction");
  const fieldLineX = document.querySelector(".field-line-x");
  const fieldLineY = document.querySelector(".field-line-y");
  const loadingStep = document.getElementById("loading-step");
  let loadingTimer = null;

  function loadHerdSize() {
    if (!herdInput) return;
    try {
      const farmer = window.HerdLineHistory?.getFarmer();
      if (farmer?.herdSize > 0) {
        herdInput.value = String(farmer.herdSize);
        return;
      }
      const saved = localStorage.getItem(STORAGE_KEY);
      if (saved) {
        const n = parseInt(saved, 10);
        if (n > 0) herdInput.value = String(n);
      }
    } catch (_) {}
  }

  function saveHerdSize() {
    if (!herdInput) return;
    const n = parseInt(herdInput.value, 10);
    if (n > 0) {
      try {
        localStorage.setItem(STORAGE_KEY, String(n));
      } catch (_) {}
    }
  }

  function syncLineFields() {
    if (!lineMode || !fieldLineX || !fieldLineY) return;
    const vertical = lineMode.value === "vertical";
    fieldLineX.classList.toggle("hidden", !vertical);
    fieldLineY.classList.toggle("hidden", vertical);
    syncDirectionOptions();
  }

  function syncDirectionOptions() {
    if (!lineMode || !directionSelect) return;
    const vertical = lineMode.value === "vertical";
    const opts = directionSelect.options;
    for (let i = 0; i < opts.length; i++) {
      const v = opts[i].value;
      const isVert = v === "left_to_right" || v === "right_to_left";
      const isHoriz = v === "down" || v === "up";
      opts[i].hidden = vertical ? !isVert : !isHoriz;
    }
    const current = directionSelect.value;
    const valid = vertical
      ? current === "left_to_right" || current === "right_to_left"
      : current === "down" || current === "up";
    if (!valid) {
      directionSelect.value = vertical
        ? directionSelect.dataset.defaultVertical || "right_to_left"
        : directionSelect.dataset.defaultHorizontal || "down";
    }
  }

  function startLoadingSteps() {
    if (!loadingStep) return;
    let i = 0;
    loadingStep.textContent = LOADING_STEPS[0];
    loadingTimer = window.setInterval(() => {
      i = (i + 1) % LOADING_STEPS.length;
      loadingStep.textContent = LOADING_STEPS[i];
    }, 2800);
  }

  loadHerdSize();
  syncLineFields();

  if (herdInput) {
    herdInput.addEventListener("change", saveHerdSize);
    herdInput.addEventListener("blur", saveHerdSize);
  }

  if (lineMode) {
    lineMode.addEventListener("change", syncLineFields);
  }

  if (fileInput && fileLabel) {
    fileInput.addEventListener("change", function () {
      const name = fileInput.files[0]?.name;
      if (name) {
        fileLabel.textContent = name;
        fileLabel.classList.add("has-file");
      } else {
        fileLabel.textContent = "Tap to choose video";
        fileLabel.classList.remove("has-file");
      }
    });
  }

  if (form && overlay) {
    form.addEventListener("submit", function (e) {
      if (!fileInput?.files?.length) {
        e.preventDefault();
        return;
      }
      if (herdInput) {
        const n = parseInt(herdInput.value, 10);
        if (!n || n < 1) {
          e.preventDefault();
          herdInput.focus();
          return;
        }
        saveHerdSize();
      }
      overlay.classList.add("active");
      startLoadingSteps();
      if (submitBtn) submitBtn.disabled = true;
    });
  }
})();
