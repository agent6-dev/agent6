/* Click a rendered mermaid diagram to open it in a pan/zoom overlay.
   No dependencies; wheel zooms, drag pans, double-click resets, Esc or a
   backdrop click closes. Armed for every diagram Material renders, including
   after instant navigation. */
(function () {
  "use strict";

  function openOverlay(svg) {
    var overlay = document.createElement("div");
    overlay.className = "a6-zoom-overlay";
    var stage = document.createElement("div");
    stage.className = "a6-zoom-stage";
    var clone = svg.cloneNode(true);
    clone.removeAttribute("width");
    clone.style.maxWidth = "none";
    stage.appendChild(clone);
    overlay.appendChild(stage);
    document.body.appendChild(overlay);

    var scale = 1;
    var tx = 0;
    var ty = 0;
    function apply() {
      stage.style.transform =
        "translate(" + tx + "px," + ty + "px) scale(" + scale + ")";
    }

    // Start at natural size, centered by flexbox; if the diagram is wider
    // than the viewport, shrink to fit once so the whole shape is visible.
    var box = clone.viewBox && clone.viewBox.baseVal;
    if (box && box.width) {
      clone.style.width = box.width + "px";
      var fit = Math.min(
        1,
        (window.innerWidth * 0.95) / box.width,
        (window.innerHeight * 0.95) / box.height
      );
      scale = fit;
      apply();
    }

    overlay.addEventListener(
      "wheel",
      function (e) {
        e.preventDefault();
        var next = scale * (e.deltaY < 0 ? 1.15 : 1 / 1.15);
        scale = Math.min(10, Math.max(0.2, next));
        apply();
      },
      { passive: false }
    );

    var dragging = false;
    var lastX = 0;
    var lastY = 0;
    overlay.addEventListener("pointerdown", function (e) {
      dragging = true;
      lastX = e.clientX;
      lastY = e.clientY;
      overlay.setPointerCapture(e.pointerId);
    });
    overlay.addEventListener("pointermove", function (e) {
      if (!dragging) return;
      tx += e.clientX - lastX;
      ty += e.clientY - lastY;
      lastX = e.clientX;
      lastY = e.clientY;
      apply();
    });
    overlay.addEventListener("pointerup", function () {
      dragging = false;
    });

    var moved = false;
    overlay.addEventListener("pointermove", function () {
      if (dragging) moved = true;
    });
    overlay.addEventListener("click", function (e) {
      if (e.target === overlay && !moved) close();
      moved = false;
    });
    overlay.addEventListener("dblclick", function () {
      scale = 1;
      tx = 0;
      ty = 0;
      apply();
    });
    function onKey(e) {
      if (e.key === "Escape") close();
    }
    document.addEventListener("keydown", onKey);
    function close() {
      document.removeEventListener("keydown", onKey);
      overlay.remove();
    }
  }

  function arm() {
    document.querySelectorAll(".mermaid svg").forEach(function (svg) {
      if (svg.dataset.a6zoom) return;
      svg.dataset.a6zoom = "1";
      svg.style.cursor = "zoom-in";
      svg.addEventListener("click", function () {
        openOverlay(svg);
      });
    });
  }

  // Material renders mermaid asynchronously and swaps content on instant
  // navigation; watch for both.
  new MutationObserver(arm).observe(document.documentElement, {
    subtree: true,
    childList: true,
  });
  if (document.readyState !== "loading") arm();
  else document.addEventListener("DOMContentLoaded", arm);
})();
