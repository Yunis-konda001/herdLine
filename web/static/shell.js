(function () {
  const toggle = document.getElementById("sidebar-toggle");
  const shell = document.querySelector(".app-shell");
  if (toggle && shell) {
    toggle.addEventListener("click", () => {
      shell.classList.toggle("sidebar-open");
    });
  }
})();
