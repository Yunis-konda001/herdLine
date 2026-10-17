(function () {
  const tabs = document.querySelectorAll("[data-auth-tab]");
  const panels = {
    signin: document.getElementById("panel-signin"),
    signup: document.getElementById("panel-signup"),
  };
  const title = document.getElementById("auth-title");
  const subtitle = document.getElementById("auth-subtitle");
  if (!tabs.length || !panels.signin || !panels.signup) return;

  function show(name) {
    tabs.forEach((tab) => {
      const on = tab.getAttribute("data-auth-tab") === name;
      tab.classList.toggle("is-active", on);
      tab.setAttribute("aria-selected", on ? "true" : "false");
    });
    panels.signin.hidden = name !== "signin";
    panels.signup.hidden = name !== "signup";
    if (title) {
      title.textContent = name === "signup" ? "Create your farm" : "Welcome back";
    }
    if (subtitle) {
      subtitle.textContent = name === "signup"
        ? "Save your farm so you can sign in later."
        : "Sign in with the email you used to create your farm.";
    }
  }

  tabs.forEach((tab) => {
    tab.addEventListener("click", () => show(tab.getAttribute("data-auth-tab")));
  });
})();
