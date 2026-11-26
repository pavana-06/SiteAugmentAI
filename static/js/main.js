(() => {
  "use strict";

  const input = document.getElementById("image-input");
  const dropzone = document.getElementById("dropzone");
  const browseButton = document.getElementById("browse-button");
  const preview = document.getElementById("source-preview");
  const sourceMeta = document.getElementById("source-meta");
  const fileName = document.getElementById("file-name");
  const fileInfo = document.getElementById("file-info");
  const generateButton = document.getElementById("generate-button");
  const resetButton = document.getElementById("reset-button");
  const resultsGrid = document.getElementById("results-grid");
  const emptyState = document.getElementById("empty-state");
  const errorBox = document.getElementById("form-error");
  const toast = document.getElementById("toast");
  const comparisonPanel = document.getElementById("comparison-panel");
  const compareSelect = document.getElementById("compare-select");
  const compareOriginal = document.getElementById("compare-original");
  const compareAugmented = document.getElementById("compare-augmented");
  const comparisonFrame = document.getElementById("comparison-frame");
  const compareClip = document.getElementById("compare-clip");
  const compareDivider = document.getElementById("compare-divider");
  const compareRange = document.getElementById("compare-range");
  const zipButton = document.getElementById("download-zip");
  const countBadge = document.getElementById("selected-count");
  const replaceButton = document.getElementById("replace-button");
  const progress = document.getElementById("generation-progress");
  const mobileMenu = document.getElementById("mobile-menu");
  const sidebar = document.getElementById("sidebar");
  const sidebarScrim = document.getElementById("sidebar-scrim");
  const resultsSubtitle = document.getElementById("results-subtitle");
  const dropContent = document.getElementById("drop-content");
  const previewOverlay = document.getElementById("preview-overlay");
  const imageCount = document.getElementById("image-count");
  let selectedFile = null;
  let previewUrl = null;
  let activeSession = null;
  let toastTimer;

  document.querySelectorAll(".effect-control input").forEach((slider) => {
    const output = slider.parentElement.querySelector("output");
    const update = () => { output.value = Number(slider.value).toFixed(2); };
    slider.addEventListener("input", update);
    update();
  });

  function updateSelectedCount() {
    const count = document.querySelectorAll(".effect-toggle:checked").length;
    if (countBadge) countBadge.textContent = `${count} selected`;
  }
  document.querySelectorAll(".effect-toggle").forEach((toggle) => toggle.addEventListener("change", updateSelectedCount));
  updateSelectedCount();

  function showError(message) {
    if (errorBox) {
      errorBox.textContent = message;
      errorBox.hidden = false;
    }
  }
  function clearError() {
    if (errorBox) {
      errorBox.hidden = true;
      errorBox.textContent = "";
    }
  }
  function showToast(message) {
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("visible");
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.classList.remove("visible"), 2800);
  }

  function clearGeneratedWorkspace() {
    if (resultsGrid) resultsGrid.replaceChildren();
    if (emptyState) emptyState.hidden = false;
    if (resultsSubtitle) resultsSubtitle.textContent = "Waiting for an upload";
    if (comparisonPanel) comparisonPanel.hidden = true;
    if (zipButton) zipButton.hidden = true;
    activeSession = null;
  }

  function clearSourceSelection() {
    selectedFile = null;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    if (preview) {
      preview.removeAttribute("src");
      preview.hidden = true;
    }
    if (dropContent) dropContent.hidden = false;
    if (previewOverlay) previewOverlay.hidden = true;
    if (sourceMeta) sourceMeta.hidden = true;
  }

  function setFile(file) {
    clearError();
    if (!file) return;
    const hasSupportedExtension = /\.(jpe?g|png)$/i.test(file.name);
    const hasSupportedMime = ["", "image/jpeg", "image/pjpeg", "image/png"].includes(file.type);
    if (!hasSupportedExtension || !hasSupportedMime) {
      clearSourceSelection();
      clearGeneratedWorkspace();
      showError("Choose a JPG or PNG image.");
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      clearSourceSelection();
      clearGeneratedWorkspace();
      showError("This image is over the 10 MB upload limit.");
      return;
    }
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    selectedFile = file;
    previewUrl = URL.createObjectURL(file);
    if (preview) {
      preview.src = previewUrl;
      preview.hidden = false;
    }
    if (dropContent) dropContent.hidden = true;
    if (previewOverlay) previewOverlay.hidden = false;
    if (sourceMeta) sourceMeta.hidden = false;
    if (fileName) fileName.textContent = file.name;
    if (fileInfo) fileInfo.textContent = `${(file.size / 1024).toFixed(0)} KB · Loading dimensions…`;
    const dimensionsImage = new Image();
    dimensionsImage.onload = () => {
      if (dimensionsImage.naturalWidth * dimensionsImage.naturalHeight > 20_000_000) {
        clearSourceSelection();
        clearGeneratedWorkspace();
        showError("Image dimensions exceed the allowed limit.");
        return;
      }
      if (fileInfo) fileInfo.textContent = `${dimensionsImage.naturalWidth} × ${dimensionsImage.naturalHeight} px · ${(file.size / 1024).toFixed(0)} KB`;
    };
    dimensionsImage.onerror = () => {
      clearSourceSelection();
      clearGeneratedWorkspace();
      showError("The selected file is not a valid image.");
    };
    dimensionsImage.src = previewUrl;
    if (resultsGrid) resultsGrid.replaceChildren();
    if (emptyState) emptyState.hidden = false;
    if (resultsSubtitle) resultsSubtitle.textContent = "Waiting for an upload";
    if (comparisonPanel) comparisonPanel.hidden = true;
    if (zipButton) zipButton.hidden = true;
    activeSession = null;
  }

  if (input) input.addEventListener("change", () => setFile(input.files[0]));
  if (browseButton && input) browseButton.addEventListener("click", (event) => { event.stopPropagation(); input.click(); });
  if (replaceButton && input) replaceButton.addEventListener("click", (event) => { event.stopPropagation(); input.click(); });
  if (dropzone && input) {
    dropzone.addEventListener("click", () => input.click());
    dropzone.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); }
    });
    ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.add("drag-over");
    }));
    ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => {
      event.preventDefault();
      dropzone.classList.remove("drag-over");
    }));
    dropzone.addEventListener("drop", (event) => setFile(event.dataTransfer.files[0]));
  }

  function setMobileNavigation(open) {
    if (sidebar) sidebar.classList.toggle("open", open);
    if (sidebarScrim) sidebarScrim.hidden = !open;
    if (mobileMenu) mobileMenu.setAttribute("aria-expanded", String(open));
  }
  if (mobileMenu) mobileMenu.addEventListener("click", () => setMobileNavigation(!sidebar?.classList.contains("open")));
  if (sidebarScrim) sidebarScrim.addEventListener("click", () => setMobileNavigation(false));
  document.querySelectorAll(".nav-item").forEach((link) => {
    link.addEventListener("click", () => {
      document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
      link.classList.add("active");
      setMobileNavigation(false);
    });
  });

  const presetRecipes = {
    "low-light": [
      ["brightness", "intensity", 0.3],
      ["gamma", "gamma", 1.5],
      ["noise", "intensity", 0.14],
    ],
    "dust": [
      ["haze", "intensity", 0.38],
      ["contrast", "factor", 1.25],
      ["occlusion", "intensity", 0.18],
    ],
    "motion": [
      ["motion_blur", "strength", 0.45],
      ["noise", "intensity", 0.12],
    ],
    "shadow": [
      ["contrast", "factor", 1.45],
      ["gamma", "gamma", 1.2],
      ["shadow", "intensity", 0.45],
    ],
  };

  document.querySelectorAll(".preset-card").forEach((card) => {
    card.addEventListener("click", () => {
      const recipe = presetRecipes[card.dataset.preset];
      if (!recipe) return;
      document.querySelectorAll(".effect-toggle").forEach((toggle) => { toggle.checked = false; });
      recipe.forEach(([effectName, parameterName, value]) => {
        const toggle = document.querySelector(`.effect-toggle[data-effect="${effectName}"]`);
        if (!toggle) return;
        toggle.checked = true;
        const slider = toggle.closest(".effect-card").querySelector(".effect-control input");
        slider.value = String(value);
        slider.dataset.param = parameterName;
        slider.dispatchEvent(new Event("input"));
      });
      document.querySelectorAll(".preset-card").forEach((item) => item.classList.remove("active"));
      card.classList.add("active");
      updateSelectedCount();
      clearError();
      showToast(`${card.querySelector("strong").textContent} preset applied.`);
    });
  });

  function readRecipe() {
    return Array.from(document.querySelectorAll(".effect-toggle:checked")).map((toggle) => {
      const card = toggle.closest(".effect-card");
      const slider = card.querySelector(".effect-control input");
      return { name: toggle.dataset.effect, params: { [slider.dataset.param]: Number(slider.value) } };
    });
  }

  function renderComparison(results) {
    compareOriginal.src = previewUrl;
    compareSelect.replaceChildren(...results.map((result, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `Variation ${String(index + 1).padStart(2, "0")}`;
      return option;
    }));
    if (comparisonPanel) comparisonPanel.hidden = false;
    const update = () => {
      const result = results[Number(compareSelect.value)];
      compareAugmented.src = result.url;
      compareAugmented.style.width = `${comparisonFrame.clientWidth}px`;
      compareAugmented.style.height = `${comparisonFrame.clientHeight}px`;
      const position = Number(compareRange.value);
      compareClip.style.width = `${position}%`;
      compareDivider.style.left = `${position}%`;
    };
    compareSelect.onchange = update;
    compareRange.oninput = update;
    update();
  }

  window.addEventListener("resize", () => {
    if (!comparisonPanel.hidden) {
      compareAugmented.style.width = `${comparisonFrame.clientWidth}px`;
      compareAugmented.style.height = `${comparisonFrame.clientHeight}px`;
    }
  });

  function createResultCard(result, index) {
    const card = document.createElement("article");
    card.className = "result-card";
    card.style.animationDelay = `${Math.min(index * 55, 330)}ms`;

    const imageWrap = document.createElement("div");
    imageWrap.className = "result-image-wrap";
    const image = document.createElement("img");
    image.src = result.url;
    image.alt = `Augmented variation ${index + 1}`;
    image.loading = "lazy";
    const number = document.createElement("span");
    number.className = "result-number";
    number.textContent = `VAR ${String(index + 1).padStart(2, "0")}`;
    imageWrap.append(image, number);

    const info = document.createElement("div");
    info.className = "result-info";
    const labels = document.createElement("div");
    labels.className = "result-labels";
    result.labels.forEach((label) => {
      const tag = document.createElement("span");
      tag.className = "result-tag";
      tag.textContent = label;
      labels.append(tag);
    });
    const dimensions = document.createElement("div");
    dimensions.className = "metric-row";
    dimensions.innerHTML = `<span>DIMENSIONS</span><strong>${result.metrics.width} × ${result.metrics.height}</strong>`;
    const metrics = document.createElement("div");
    metrics.className = "metric-row";
    const ssim = result.metrics.ssim === null ? "—" : result.metrics.ssim.toFixed(3);
    metrics.innerHTML = `<span>SSIM&nbsp; · &nbsp;PSNR</span><strong>${ssim} · ${result.metrics.psnr === null ? "∞" : `${result.metrics.psnr} dB`}</strong>`;
    const brightness = document.createElement("div");
    brightness.className = "metric-row";
    brightness.innerHTML = `<span>MEAN BRIGHTNESS</span><strong>${result.metrics.mean_brightness.toFixed(2)}</strong>`;
    const download = document.createElement("a");
    download.className = "result-download";
    download.href = result.url;
    download.download = `siteaugment-variation-${String(index + 1).padStart(2, "0")}.png`;
    download.textContent = "↓  Download PNG";
    info.append(labels, dimensions, metrics, brightness, download);
    card.append(imageWrap, info);
    return card;
  }

  generateButton.addEventListener("click", async () => {
    clearError();
    if (!selectedFile) {
      showError("Upload an image before generating variations.");
      return;
    }
    const augmentations = readRecipe();
    if (!augmentations.length) {
      showError("Select at least one augmentation effect.");
      return;
    }
    const config = { augmentations, count: Number(imageCount?.value ?? "4") };
    const form = new FormData();
    form.append("image", selectedFile);
    form.append("config", JSON.stringify(config));
    if (generateButton) {
      generateButton.disabled = true;
      generateButton.innerHTML = '<span class="generate-icon" aria-hidden="true">✦</span><span class="generate-label">Generating…</span><span aria-hidden="true">···</span>';
    }
    if (progress) progress.hidden = false;
    try {
      const response = await fetch("/augment", { method: "POST", body: form });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || "Could not generate images.");
      activeSession = payload.session_id;
      resultsGrid.replaceChildren(...payload.results.map(createResultCard));
      if (emptyState) emptyState.hidden = true;
      if (resultsSubtitle) resultsSubtitle.textContent = `${payload.results.length} variations generated · ${payload.original.width} × ${payload.original.height} px`;
      if (zipButton) zipButton.hidden = false;
      renderComparison(payload.results);
      showToast(`${payload.results.length} augmented images generated.`);
    } catch (error) {
      showError(error.message || "Something went wrong while generating images.");
    } finally {
      if (generateButton) {
        generateButton.disabled = false;
        generateButton.innerHTML = '<span class="generate-icon" aria-hidden="true">✦</span><span class="generate-label">Generate images</span><span aria-hidden="true">→</span>';
      }
      if (progress) progress.hidden = true;
    }
  });

  if (zipButton) zipButton.addEventListener("click", () => {
    if (activeSession) window.location.href = `/download-zip/${encodeURIComponent(activeSession)}`;
  });

  if (resetButton) resetButton.addEventListener("click", () => {
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    selectedFile = null;
    activeSession = null;
    input.value = "";
    if (preview) {
      preview.removeAttribute("src");
      preview.hidden = true;
    }
    if (dropContent) dropContent.hidden = false;
    if (previewOverlay) previewOverlay.hidden = true;
    if (sourceMeta) sourceMeta.hidden = true;
    if (resultsGrid) resultsGrid.replaceChildren();
    if (emptyState) emptyState.hidden = false;
    if (comparisonPanel) comparisonPanel.hidden = true;
    if (zipButton) zipButton.hidden = true;
    if (resultsSubtitle) resultsSubtitle.textContent = "Waiting for an upload";
    if (imageCount) imageCount.value = "4";
    document.querySelectorAll(".effect-toggle").forEach((toggle) => { toggle.checked = toggle.dataset.effect === "brightness"; });
    document.querySelectorAll(".effect-control input").forEach((slider) => {
      slider.value = slider.defaultValue;
      slider.dispatchEvent(new Event("input"));
    });
    compareRange.value = "50";
    document.querySelectorAll(".preset-card").forEach((card) => card.classList.remove("active"));
    updateSelectedCount();
    clearError();
    showToast("Workspace reset.");
  });
})();
