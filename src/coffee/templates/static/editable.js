document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("[data-mlcedit]").forEach(function (el) {
    const isEditable = el.isContentEditable;
    el.contentEditable = !isEditable;
  });

  function updateURL() {
    const edits = {};
    document.querySelectorAll("[data-mlcedit]").forEach((el) => {
      const id = el.getAttribute("data-mlcedit");
      edits[id] = encodeURIComponent(el.innerHTML);
    });
    const newUrl = `${location.pathname}?${new URLSearchParams(
      edits,
    ).toString()}`;
    history.pushState({}, "", newUrl);
  }

  document.querySelectorAll("[data-mlcedit]").forEach((el) => {
    ["blur", "change", "keyup"].forEach((event) => {
      el.addEventListener(event, updateURL);
    });
  });

  function loadEdits() {
    const params = new URLSearchParams(window.location.search);
    params.forEach((value, key) => {
      const el = document.querySelector(`[data-mlcedit="${key}"]`);
      if (el) {
        el.innerHTML = decodeURIComponent(value);
      }
    });
  }

  loadEdits();
});
