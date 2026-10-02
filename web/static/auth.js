(function () {
  function initials(farmerName, farmName) {
    const src = (farmerName || farmName || "?").trim();
    const parts = src.split(/\s+/).filter(Boolean);
    if (parts.length >= 2) {
      return (parts[0][0] + parts[1][0]).toUpperCase();
    }
    return src.slice(0, 2).toUpperCase();
  }

  function bindProfilePreview() {
    const farm = document.getElementById("farm_name");
    const person = document.getElementById("farmer_name");
    const contact = document.getElementById("contact");
    const herd = document.getElementById("profile_herd_size");
    if (!farm) return;

    function paint() {
      const farmVal = farm.value.trim() || "Farm name";
      const personVal = person?.value.trim() || "Farmer name";
      const contactVal = contact?.value.trim() || "";
      const herdVal = herd?.value.trim() || "—";

      const elFarm = document.getElementById("preview-farm");
      const elPerson = document.getElementById("preview-farmer");
      const elContact = document.getElementById("preview-contact");
      const elHerd = document.getElementById("preview-herd");
      const elAvatar = document.getElementById("preview-avatar");
      const elCheckins = document.getElementById("preview-checkins");

      if (elFarm) elFarm.textContent = farmVal;
      if (elPerson) elPerson.textContent = personVal;
      const elContactHidden = document.getElementById("preview-contact");
      if (elContactHidden) elContactHidden.textContent = contactVal;
      if (elHerd) elHerd.textContent = herdVal;
      if (elAvatar) elAvatar.textContent = initials(personVal, farmVal);
      if (elCheckins && window.HerdLineHistory) {
        elCheckins.textContent = String(window.HerdLineHistory.getCheckins().length);
      }
    }

    [farm, person, contact, herd].forEach((el) => {
      el?.addEventListener("input", paint);
    });
    paint();
  }

  function prefillSignup() {
    const farmer = window.HerdLineHistory?.getFarmer();
    if (!farmer) return;
    const map = {
      farm_name: farmer.farmName,
      farmer_name: farmer.farmerName,
      contact: farmer.contact,
      profile_herd_size: farmer.herdSize,
      location_note: farmer.locationNote,
    };
    Object.entries(map).forEach(([id, val]) => {
      const el = document.getElementById(id);
      if (el && val != null && val !== "") el.value = val;
    });
  }

  const signupForm = document.getElementById("signup-form");
  if (signupForm) {
    prefillSignup();
    bindProfilePreview();
    signupForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const herd = parseInt(document.getElementById("profile_herd_size").value, 10);
      if (!herd || herd < 1) return;
      const profile = {
        farmName: document.getElementById("farm_name").value.trim(),
        farmerName: document.getElementById("farmer_name").value.trim(),
        contact: document.getElementById("contact").value.trim(),
        herdSize: herd,
        locationNote: document.getElementById("location_note").value.trim(),
        createdAt: new Date().toISOString(),
      };
      window.HerdLineHistory.saveFarmer(profile);
      window.location.href = "/dashboard";
    });
  }

  const loginForm = document.getElementById("login-form");
  const existing = document.getElementById("login-existing");
  const loginFooter = document.getElementById("login-form-footer");
  const continueBtn = document.getElementById("login-continue");
  if (loginForm) {
    const farmer = window.HerdLineHistory?.getFarmer();
    if (farmer?.farmName && existing) {
      existing.hidden = false;
      loginForm.hidden = true;
      if (loginFooter) loginFooter.hidden = true;
      document.getElementById("login-farm-name").textContent = farmer.farmName;
      const farmerEl = document.getElementById("login-farmer-name");
      if (farmerEl) {
        farmerEl.textContent = farmer.farmerName || farmer.contact || "";
        farmerEl.hidden = !(farmer.farmerName || farmer.contact);
      }
      const av = document.getElementById("login-avatar");
      if (av) av.textContent = initials(farmer.farmerName, farmer.farmName);
      continueBtn?.addEventListener("click", () => {
        window.location.href = "/dashboard";
      });
    }
    loginForm.addEventListener("submit", (e) => {
      e.preventDefault();
      const name = document.getElementById("login_farm_name").value.trim();
      const stored = window.HerdLineHistory?.getFarmer();
      if (stored && stored.farmName === name) {
        window.location.href = "/dashboard";
        return;
      }
      if (stored) {
        alert("Farm name does not match this device. Open Profile to update.");
        return;
      }
      alert("No profile on this device yet. Create a profile first.");
    });
  }
})();
