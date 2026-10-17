(function () {
  const toggle = document.getElementById("sidebar-toggle");
  const shell = document.querySelector(".app-shell");
  if (toggle && shell) {
    toggle.addEventListener("click", () => {
      shell.classList.toggle("sidebar-open");
    });
  }

  function clearLocalProfile() {
    try {
      localStorage.removeItem("herdline_farmer");
    } catch (_) {
      /* the server session is still cleared */
    }
  }

  document.getElementById("logout-form")?.addEventListener("submit", clearLocalProfile);
  document.getElementById("logout-form-mobile")?.addEventListener("submit", clearLocalProfile);
  document.getElementById("logout-form-dash")?.addEventListener("submit", clearLocalProfile);
})();
