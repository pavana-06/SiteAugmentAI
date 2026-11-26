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
  const confidenceInput = document.getElementById("detection-threshold");
  const confidenceOutput = document.getElementById("detection-threshold-value");
  const detectOriginalButton = document.getElementById("detect-original");
  const detectionSection = document.getElementById("detection-section");
  const detectionGrid = document.getElementById("detection-grid");
  const detectionComparison = document.getElementById("detection-comparison");
  const originalDetectionCount = document.getElementById("original-detection-count");
  const augmentedDetectionCount = document.getElementById("augmented-detection-count");
  let selectedFile = null;
  let previewUrl = null;
  let activeSession = null;
  const detectionResults = new Map();
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

  function clearDetectionResults() {
    detectionResults.clear();
    if (detectionGrid) detectionGrid.replaceChildren();
    if (detectionSection) detectionSection.hidden = true;
    if (detectionComparison) detectionComparison.hidden = true;
  }

  function clearAugmentedDetectionResults() {
    detectionResults.forEach((result, key) => {
      if (result.source !== "augmented") return;
      detectionResults.delete(key);
      const card = [...(detectionGrid?.querySelectorAll(".detection-card") || [])]
        .find((item) => item.dataset.detectionKey === key);
      if (card) card.remove();
    });
    if (detectionSection) detectionSection.hidden = detectionResults.size === 0;
    updateDetectionComparison();
  }

  function clearGeneratedWorkspace() {
    if (resultsGrid) resultsGrid.replaceChildren();
    if (emptyState) emptyState.hidden = false;
    if (resultsSubtitle) resultsSubtitle.textContent = "Waiting for an upload";
    if (comparisonPanel) comparisonPanel.hidden = true;
    if (zipButton) zipButton.hidden = true;
    clearDetectionResults();
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
    if (detectOriginalButton) detectOriginalButton.disabled = true;
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
    if (detectOriginalButton) detectOriginalButton.disabled = false;
    const dimensionsImage = new Image();
    dimensionsImage.onload = () => {
      if (dimensionsImage.naturalWidth * dimensionsImage.naturalHeight > 20_000_000) {
        clearSourceSelection();
        clearGeneratedWorkspace();
        if (detectOriginalButton) detectOriginalButton.disabled = true;
        showError("Image dimensions exceed the allowed limit.");
        return;
      }
      if (fileInfo) fileInfo.textContent = `${dimensionsImage.naturalWidth} × ${dimensionsImage.naturalHeight} px · ${(file.size / 1024).toFixed(0)} KB`;
    };
    dimensionsImage.onerror = () => {
      clearSourceSelection();
      clearGeneratedWorkspace();
      if (detectOriginalButton) detectOriginalButton.disabled = true;
      showError("The selected file is not a valid image.");
    };
    dimensionsImage.src = previewUrl;
    if (resultsGrid) resultsGrid.replaceChildren();
    if (emptyState) emptyState.hidden = false;
    if (resultsSubtitle) resultsSubtitle.textContent = "Waiting for an upload";
    if (comparisonPanel) comparisonPanel.hidden = true;
    if (zipButton) zipButton.hidden = true;
    clearDetectionResults();
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
    "balanced": [
      ["brightness", "intensity", 0.12],
      ["contrast", "factor", 1.1],
      ["shadow", "intensity", 0.12],
      ["noise", "intensity", 0.04],
      ["haze", "intensity", 0.08],
    ],
    "low-light": [
      ["brightness", "intensity", 0.15],
      ["gamma", "gamma", 1.25],
      ["noise", "intensity", 0.05],
    ],
    "dust": [
      ["haze", "intensity", 0.14],
      ["contrast", "factor", 1.1],
      ["occlusion", "intensity", 0.08],
    ],
    "motion": [
      ["motion_blur", "strength", 0.15],
      ["noise", "intensity", 0.05],
    ],
    "shadow": [
      ["contrast", "factor", 1.2],
      ["gamma", "gamma", 1.1],
      ["shadow", "intensity", 0.18],
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

  function resultPreviewSource(result) {
    return result.preview_data_url || result.preview_url || result.url;
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
      compareAugmented.src = resultPreviewSource(result);
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
    if (comparisonPanel && !comparisonPanel.hidden) {
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
    image.alt = `Augmented variation ${index + 1}`;
    image.loading = "eager";
    image.addEventListener("error", () => {
      if (result.preview_data_url && !image.dataset.previewFallbackAttempted) {
        image.dataset.previewFallbackAttempted = "true";
        image.src = result.preview_url || result.url;
        return;
      }
      const error = document.createElement("span");
      error.className = "result-preview-error";
      error.textContent = "Preview unavailable";
      image.replaceWith(error);
    });
    image.src = resultPreviewSource(result);
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
    const detectButton = document.createElement("button");
    detectButton.className = "result-detect-button";
    detectButton.type = "button";
    detectButton.textContent = "⌖ Detect PPE";
    detectButton.addEventListener("click", () => detectAugmented(result, detectButton));
    info.append(labels, dimensions, metrics, brightness, download, detectButton);
    card.append(imageWrap, info);
    return card;
  }

  function createDetectionCard(result, key) {
    const card = document.createElement("article");
    card.className = "detection-card";
    card.dataset.detectionKey = key;
    const heading = document.createElement("div");
    heading.className = "detection-card-heading";
    const title = document.createElement("div");
    const sourceTitle = result.source === "original"
      ? "Original image"
      : `Augmented image · ${result.source_index}`;
    const headingTitle = document.createElement("strong");
    headingTitle.textContent = sourceTitle;
    const dimensions = document.createElement("small");
    dimensions.textContent = `${result.image.width} × ${result.image.height} px · threshold ${result.confidence_threshold.toFixed(2)}`;
    title.append(headingTitle, dimensions);
    const total = document.createElement("span");
    total.className = "detection-total";
    total.textContent = `${result.summary.total_detections} detected`;
    heading.append(title, total);

    const visual = document.createElement("div");
    visual.className = "detection-visual";
    const image = document.createElement("img");
    image.alt = `PPE detection visualization: ${sourceTitle}`;
    image.loading = "eager";
    let usingInlinePreview = Boolean(result.annotated_preview_data_url);
    image.addEventListener("error", () => {
      if (usingInlinePreview && result.annotated_url) {
        usingInlinePreview = false;
        image.src = result.annotated_url;
        return;
      }
      const error = document.createElement("span");
      error.className = "detection-image-error";
      error.setAttribute("role", "status");
      error.textContent = "Visualization could not be loaded.";
      image.replaceWith(error);
    });
    image.src = result.annotated_preview_data_url || result.annotated_url;
    visual.append(image);

    const body = document.createElement("div");
    body.className = "detection-card-body";
    const counts = document.createElement("div");
    counts.className = "class-counts";
    Object.entries(result.summary.class_counts).forEach(([label, count]) => {
      const chip = document.createElement("span");
      chip.className = "class-count-chip";
      chip.append(document.createTextNode(label));
      const value = document.createElement("b");
      value.textContent = String(count);
      chip.append(value);
      counts.append(chip);
    });

    const confidence = document.createElement("div");
    confidence.className = "confidence-summary";
    const distribution = result.summary.confidence;
    confidence.textContent = distribution.mean === null
      ? "CONFIDENCE · No objects above threshold"
      : `CONFIDENCE · min ${distribution.min.toFixed(2)} · mean ${distribution.mean.toFixed(2)} · max ${distribution.max.toFixed(2)}`;
    body.append(counts, confidence);
    if (result.summary.total_detections === 0) {
      const noDetections = document.createElement("p");
      noDetections.className = "detection-empty-state";
      noDetections.textContent = "No objects exceeded the confidence threshold.";
      body.append(noDetections);
    }

    const boxes = document.createElement("div");
    boxes.className = "box-list";
    result.detections.forEach((detection) => {
      const row = document.createElement("div");
      row.className = "box-row";
      const label = document.createElement("span");
      label.textContent = detection.label;
      const score = document.createElement("span");
      score.textContent = `${(detection.confidence * 100).toFixed(1)}%`;
      row.append(label, score);
      boxes.append(row);
    });

    const download = document.createElement("a");
    download.className = "detection-download";
    download.href = result.annotated_download_url;
    download.download = `siteaugment-detections-${result.annotated_filename}`;
    download.textContent = "↓ Download annotated PNG";
    body.append(boxes, download);
    card.append(heading, visual, body);
    return card;
  }

  function updateDetectionComparison() {
    if (!detectionComparison || !originalDetectionCount || !augmentedDetectionCount) return;
    const original = detectionResults.get("original");
    const augmented = [...detectionResults.values()].filter((item) => item.source === "augmented");
    if (!original || augmented.length === 0) {
      detectionComparison.hidden = true;
      return;
    }
    const mean = augmented.reduce((sum, item) => sum + item.summary.total_detections, 0) / augmented.length;
    originalDetectionCount.textContent = String(original.summary.total_detections);
    augmentedDetectionCount.textContent = mean.toFixed(1);
    detectionComparison.hidden = false;
  }

  function showDetectionResult(result, key) {
    if (!detectionGrid || !detectionSection) return;
    const prior = [...detectionGrid.querySelectorAll(".detection-card")]
      .find((card) => card.dataset.detectionKey === key);
    if (prior) prior.remove();
    detectionResults.set(key, result);
    detectionGrid.append(createDetectionCard(result, key));
    detectionSection.hidden = false;
    updateDetectionComparison();
  }

  function confidenceThreshold() {
    return confidenceInput ? Number(confidenceInput.value) : 0.25;
  }

  async function readJsonResponse(response, operation) {
    const body = await response.text();
    let payload;
    try {
      payload = JSON.parse(body);
    } catch {
      const routingHint = response.status === 404
        ? " Check that deployment routing sends this request to Flask."
        : "";
      throw new Error(
        `${operation} returned a non-JSON response (HTTP ${response.status}).${routingHint}`
      );
    }
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) {
      throw new Error(`${operation} returned an invalid response.`);
    }
    if (!response.ok) {
      throw new Error(payload.error || `${operation} failed (HTTP ${response.status}).`);
    }
    return payload;
  }

  if (confidenceInput && confidenceOutput) {
    const updateConfidence = () => { confidenceOutput.value = Number(confidenceInput.value).toFixed(2); };
    confidenceInput.addEventListener("input", updateConfidence);
    updateConfidence();
  }

  async function submitDetection(button, action, body) {
    clearError();
    const previousText = button.textContent;
    button.disabled = true;
    button.textContent = "Loading model and detecting…";
    try {
      const response = await fetch(action, body);
      return await readJsonResponse(response, "PPE detection");
    } finally {
      button.disabled = false;
      button.textContent = previousText;
    }
  }

  if (detectOriginalButton) detectOriginalButton.addEventListener("click", async () => {
    if (!selectedFile) {
      showError("Upload an image before running PPE detection.");
      return;
    }
    const form = new FormData();
    form.append("image", selectedFile);
    form.append("confidence", String(confidenceThreshold()));
    try {
      const result = await submitDetection(detectOriginalButton, "/detect", {
        method: "POST",
        body: form,
      });
      result.source_index = "original";
      result.annotated_download_url = result.annotated_url.replace("/preview/", "/download/");
      showDetectionResult(result, "original");
      showToast(`${result.summary.total_detections} PPE objects detected in original image.`);
    } catch (error) {
      showError(error.message || "PPE detection failed.");
    }
  });

  async function detectAugmented(result, button) {
    if (!activeSession) {
      showError("Generate an augmentation session before detecting this image.");
      return;
    }
    const previousText = button.textContent;
    button.disabled = true;
    button.textContent = "Detecting…";
    try {
      const response = await fetch("/detect-augmented", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: activeSession,
          filename: result.filename,
          confidence: confidenceThreshold(),
        }),
      });
      const payload = await readJsonResponse(response, "PPE detection");
      payload.source_index = String([...resultsGrid.children].findIndex((card) =>
        card.querySelector(".result-detect-button") === button) + 1).padStart(2, "0");
      payload.annotated_download_url = payload.annotated_url.replace("/preview/", "/download/");
      showDetectionResult(payload, `augmented:${result.filename}`);
      showToast(`${payload.summary.total_detections} PPE objects detected in variation.`);
    } catch (error) {
      showError(error.message || "PPE detection failed.");
    } finally {
      button.disabled = false;
      button.textContent = previousText;
    }
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
      const payload = await readJsonResponse(response, "Image generation");
      activeSession = payload.session_id;
      clearAugmentedDetectionResults();
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
    if (detectOriginalButton) detectOriginalButton.disabled = true;
    clearDetectionResults();
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
