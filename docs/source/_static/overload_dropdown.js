/**
 * For each injected `.overloads-block` panel, find the function-signature
 * `<dt>` that owns it and insert an "Overloads ▶" toggle button into it.
 *
 * Layout strategy
 * ───────────────
 * Both the button and the existing "[source]" link use `float: right`.
 * Inserting the button *before* "[source]" in the DOM, combined with
 * `clear: right` on the "[source]" float (applied via CSS), causes them to
 * stack vertically rather than sit side by side.
 *
 * `display: flow-root` is set on `<dt>` so it establishes a block formatting
 * context and expands to contain both floated elements.  Functions that have
 * no overloads are never touched.
 */
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll(".overloads-block").forEach(function (block) {
    // The block lives inside <dd>, which is a child of <dl class="py …">.
    var dl = block.closest("dl.py");
    if (!dl) return;

    // Use :scope to get a *direct* child <dt> (avoids matching nested dls).
    var dt = dl.querySelector(":scope > dt");
    if (!dt) return;

    // Start collapsed.
    block.hidden = true;

    // Build the button.
    var btn = document.createElement("button");
    btn.type = "button";
    btn.className = "overloads-toggle";
    btn.setAttribute("aria-expanded", "false");

    var label = document.createElement("span");
    label.textContent = "Overloads";

    var chevron = document.createElement("span");
    chevron.className = "overloads-toggle-chevron";
    chevron.setAttribute("aria-hidden", "true");

    btn.appendChild(label);
    btn.appendChild(chevron);

    // Insert immediately before "[source]" so CSS `clear: right` on the
    // viewcode-link makes it wrap below the button.
    // When there is no "[source]" link, append at the end instead.
    var sourceLink = dt.querySelector(".viewcode-link")?.closest("a");
    dt.insertBefore(btn, sourceLink ?? null);

    // Make <dt> contain both floated elements (button + [source]).
    dt.style.display = "flow-root";

    btn.addEventListener("click", function () {
      var opening = block.hidden;
      block.hidden = !opening;
      btn.setAttribute("aria-expanded", opening ? "true" : "false");
      btn.classList.toggle("overloads-toggle--open", opening);
    });
  });
});
