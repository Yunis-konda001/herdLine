(function (global) {
  const CHECKINS_KEY = "herdline_checkins";
  const FARMER_KEY = "herdline_farmer";
  const HERD_KEY = "herdline_herd_size";
  const MAX_CHECKINS = 30;

  function safeParse(raw, fallback) {
    try {
      return JSON.parse(raw);
    } catch (_) {
      return fallback;
    }
  }

  function getFarmer() {
    try {
      const raw = localStorage.getItem(FARMER_KEY);
      if (!raw) return null;
      const f = safeParse(raw, null);
      return f && typeof f === "object" ? f : null;
    } catch (_) {
      return null;
    }
  }

  function saveFarmer(profile) {
    localStorage.setItem(FARMER_KEY, JSON.stringify(profile));
    if (profile.herdSize > 0) {
      localStorage.setItem(HERD_KEY, String(profile.herdSize));
    }
  }

  function getCheckins() {
    try {
      const raw = localStorage.getItem(CHECKINS_KEY);
      const list = safeParse(raw, []);
      return Array.isArray(list) ? list : [];
    } catch (_) {
      return [];
    }
  }

  function setCheckins(list) {
    localStorage.setItem(CHECKINS_KEY, JSON.stringify(list));
  }

  function saveCheckin(record) {
    if (!record || !record.id) return;
    const farmer = getFarmer();
    if (farmer && !record.farmName) {
      record.farmName = farmer.farmName || farmer.farmerName || "My farm";
    }
    let list = getCheckins().filter((c) => c.id !== record.id);
    list.unshift(record);
    if (list.length > MAX_CHECKINS) {
      list = list.slice(0, MAX_CHECKINS);
    }
    setCheckins(list);
  }

  function getCheckin(id) {
    return getCheckins().find((c) => c.id === id) || null;
  }

  function deleteCheckin(id) {
    if (!id) return;
    const ok = window.confirm(
      "Remove this check-in from your history on this device? The summary will be deleted; the video file may still remain on the server."
    );
    if (!ok) return;
    setCheckins(getCheckins().filter((c) => c.id !== id));
    const detailRoot = document.getElementById("checkin-detail-root");
    if (detailRoot?.getAttribute("data-checkin-id") === id) {
      window.location.href = "/checkins";
      return;
    }
    refreshAllHistoryViews();
  }

  function refreshAllHistoryViews() {
    initDashboard();
    paintCheckinsList();
    const previewCheckins = document.getElementById("preview-checkins");
    if (previewCheckins) {
      previewCheckins.textContent = String(getCheckins().length);
    }
  }

  let checkinsListFilter = "all";
  let checkinsFiltersBound = false;

  function startOfDay(d) {
    const x = new Date(d);
    x.setHours(0, 0, 0, 0);
    return x;
  }

  function formatDateLabel(iso) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "Unknown date";
    const today = startOfDay(new Date());
    const that = startOfDay(d);
    const diffDays = Math.round((today - that) / 86400000);
    if (diffDays === 0) return "Today";
    if (diffDays === 1) return "Yesterday";
    return d.toLocaleDateString(undefined, {
      weekday: "short",
      month: "short",
      day: "numeric",
    });
  }

  function statusLabel(status) {
    if (status === "complete") return "Complete";
    if (status === "missing") return "Missing";
    if (status === "extra") return "Review";
    return "Unknown";
  }

  function escapeHtml(s) {
    const div = document.createElement("div");
    div.textContent = s;
    return div.innerHTML;
  }

  function renderStatusBadge(status) {
    return `<span class="badge badge-${escapeHtml(status)}">${escapeHtml(statusLabel(status))}</span>`;
  }

  function renderSummaryHero(c, opts) {
    const compact = opts && opts.compact;
    const status = c.status || "unknown";
    let headline = "Counting complete";
    if (status === "complete") headline = "All accounted for";
    else if (status === "missing") {
      const n = c.missing ?? 0;
      headline = `${n} goat${n !== 1 ? "s" : ""} may be missing`;
    } else if (status === "extra") headline = "Extra goats detected";

    const missing = c.missing ?? 0;
    const extra = c.extra ?? 0;
    const accentLabel = status === "extra" ? "Extra" : "Missing";
    const accentVal =
      status === "extra" ? extra : status === "missing" ? missing : 0;

    let lede = `${c.counted} goats counted in this video.`;
    if (c.herdSize != null) {
      lede = `${c.counted} counted of ${c.herdSize} in your herd.`;
      if (status === "missing") lede += ` ${missing} may be missing.`;
      else if (status === "extra") lede += ` ${extra} over your herd size.`;
    }

    const acc =
      c.accuracy != null
        ? `<p class="summary-accuracy">Match accuracy: ${Number(c.accuracy).toFixed(1)}%</p>`
        : "";

    const video =
      c.videoUrl && !compact
        ? `<section class="card"><h2 class="card-heading">Annotated video</h2><div class="video-wrap"><video controls playsinline src="${escapeHtml(c.videoUrl)}"></video></div></section>`
        : "";

    const compactClass = compact ? " summary-hero--compact" : "";

    return `
      <section class="summary-hero status-${escapeHtml(status)}${compactClass}">
        <p class="summary-date-label">${escapeHtml(formatDateLabel(c.at))} · ${escapeHtml(new Date(c.at).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }))}</p>
        <h1 class="summary-headline">${escapeHtml(headline)}</h1>
        <p class="summary-lede">${lede}</p>
        <div class="summary-numbers">
          <div class="summary-num"><span class="summary-num-label">Counted</span><span class="summary-num-value">${c.counted}</span></div>
          ${
            c.herdSize != null
              ? `<div class="summary-num"><span class="summary-num-label">Your herd</span><span class="summary-num-value">${c.herdSize}</span></div>
          <div class="summary-num summary-num-accent"><span class="summary-num-label">${accentLabel}</span><span class="summary-num-value">${accentVal}</span></div>`
              : ""
          }
        </div>
        ${acc}
      </section>
      ${video}
    `;
  }

  function bindCheckinTable(tbody) {
    tbody.querySelectorAll(".checkin-row").forEach((row) => {
      row.tabIndex = 0;
      row.setAttribute("role", "link");
    });
    if (tbody.dataset.checkinBound === "1") return;
    tbody.dataset.checkinBound = "1";

    tbody.addEventListener("click", (e) => {
      const deleteBtn = e.target.closest("[data-delete-id]");
      if (deleteBtn) {
        e.preventDefault();
        e.stopPropagation();
        deleteCheckin(deleteBtn.getAttribute("data-delete-id"));
        return;
      }
      const row = e.target.closest(".checkin-row");
      const href = row?.getAttribute("data-href");
      if (href) window.location.href = href;
    });

    tbody.addEventListener("keydown", (e) => {
      if (e.key !== "Enter" && e.key !== " ") return;
      if (e.target.closest(".btn-delete")) return;
      const row = e.target.closest(".checkin-row");
      const href = row?.getAttribute("data-href");
      if (href) {
        e.preventDefault();
        window.location.href = href;
      }
    });
  }

  function renderCheckinRow(c) {
    const missing = c.missing ?? 0;
    const sub =
      c.status === "missing"
        ? `${missing} missing`
        : c.status === "extra"
          ? `${c.extra ?? 0} extra`
          : "All accounted for";
    return `
      <tr class="checkin-row" data-href="/checkins/${encodeURIComponent(c.id)}">
        <td><span class="date-primary">${escapeHtml(formatDateLabel(c.at))}</span><span class="date-secondary">${escapeHtml(new Date(c.at).toLocaleTimeString(undefined, { hour: "numeric", minute: "2-digit" }))}</span></td>
        <td>${c.counted} / ${c.herdSize ?? "—"}</td>
        <td>${escapeHtml(sub)}</td>
        <td>${renderStatusBadge(c.status)}</td>
        <td class="checkin-actions">
          <button type="button" class="btn-delete" data-delete-id="${escapeHtml(c.id)}" aria-label="Delete check-in" title="Delete">×</button>
        </td>
      </tr>`;
  }

  function paintCheckinsList() {
    const tbody = document.getElementById("checkins-list-body");
    if (!tbody) return;
    let list = getCheckins();
    if (checkinsListFilter !== "all") {
      list = list.filter((c) => c.status === checkinsListFilter);
    }
    if (!list.length) {
      tbody.innerHTML = `<tr><td colspan="5" class="empty-cell">No check-ins match this filter.</td></tr>`;
      return;
    }
    tbody.innerHTML = list.map(renderCheckinRow).join("");
    bindCheckinTable(tbody);
  }

  function initDashboard() {
    const root = document.getElementById("dashboard-root");
    if (!root) return;
    const farmer = getFarmer();
    const greetEl = document.getElementById("dashboard-greeting-name");
    const subEl = document.getElementById("dashboard-farm-name");
    const first =
      farmer?.farmerName?.trim().split(/\s+/)[0] ||
      farmer?.farmName?.trim().split(/\s+/)[0] ||
      "there";
    if (greetEl) greetEl.textContent = first;
    if (subEl) {
      subEl.textContent = farmer?.farmName
        ? `${farmer.farmName} · latest gate counts on this device`
        : "Set up your farm profile · counts stay on this device";
    }
    const list = getCheckins();
    const last = list[0];
    const lastEl = document.getElementById("dashboard-last-checkin");
    if (lastEl) {
      if (!last) {
        lastEl.innerHTML = `<div class="empty-state"><p>No check-ins yet.</p><a class="btn-primary" href="/">Run your first check-in</a></div>`;
      } else {
        lastEl.innerHTML = renderSummaryHero(last, { compact: true });
        lastEl.insertAdjacentHTML(
          "beforeend",
          `<a class="summary-open-link" href="/checkins/${encodeURIComponent(last.id)}">Open full summary →</a>`
        );
      }
    }
    const tbody = document.getElementById("dashboard-checkins-body");
    if (tbody) {
      if (!list.length) {
        tbody.innerHTML = `<tr><td colspan="5" class="empty-cell">Check-ins appear here after each count.</td></tr>`;
      } else {
        tbody.innerHTML = list.slice(0, 5).map(renderCheckinRow).join("");
        bindCheckinTable(tbody);
      }
    }
    const demo = new URLSearchParams(window.location.search).get("demo") === "1";
    const tips = document.getElementById("presenter-tips");
    if (tips) tips.hidden = !demo;
  }

  function initCheckinsList() {
    const tbody = document.getElementById("checkins-list-body");
    if (!tbody) return;
    if (!checkinsFiltersBound) {
      checkinsFiltersBound = true;
      const chips = document.querySelectorAll("[data-filter]");
      chips.forEach((chip) => {
        chip.addEventListener("click", () => {
          checkinsListFilter = chip.getAttribute("data-filter") || "all";
          chips.forEach((c) => c.classList.toggle("active", c === chip));
          paintCheckinsList();
        });
      });
    }
    paintCheckinsList();
  }

  function initCheckinDetail() {
    const root = document.getElementById("checkin-detail-root");
    const id = root?.getAttribute("data-checkin-id");
    if (!root || !id) return;
    const c = getCheckin(id);
    if (!c) {
      root.innerHTML = `<div class="empty-state"><p>Check-in not found on this device.</p><a href="/checkins">View all check-ins</a></div>`;
      return;
    }
    root.innerHTML = renderSummaryHero(c);
    const actions = document.createElement("div");
    actions.className = "detail-actions";
    actions.innerHTML = `
      <a class="btn-primary" href="/">New check-in</a>
      <a class="btn-secondary" href="/dashboard">Dashboard</a>
      <button type="button" class="btn-secondary btn-delete-text" data-delete-id="${escapeHtml(c.id)}">Delete check-in</button>`;
    root.appendChild(actions);
    actions.querySelector("[data-delete-id]")?.addEventListener("click", () => {
      deleteCheckin(c.id);
    });
  }

  function saveFromPayloadElement() {
    const el = document.getElementById("checkin-payload");
    if (!el) return;
    const data = safeParse(el.textContent, null);
    if (!data) return;
    saveCheckin(data);
  }

  function initials(farmerName, farmName) {
    const src = (farmerName || farmName || "H").trim();
    const parts = src.split(/\s+/).filter(Boolean);
    if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase();
    return src.slice(0, 2).toUpperCase();
  }

  function updateNavAccount() {
    const farmer = getFarmer();
    const sidebarAvatar = document.getElementById("sidebar-avatar");
    const sidebarFarmer = document.getElementById("sidebar-farmer-name");
    const sidebarFarm = document.getElementById("sidebar-farm-name");
    const profileLink = document.getElementById("nav-profile");

    if (farmer?.farmName) {
      if (sidebarFarmer) {
        sidebarFarmer.textContent = farmer.farmerName || "Farmer";
      }
      if (sidebarFarm) sidebarFarm.textContent = farmer.farmName;
      if (sidebarAvatar) {
        sidebarAvatar.textContent = initials(farmer.farmerName, farmer.farmName);
      }
      if (profileLink) profileLink.textContent = "Farm profile";
    } else {
      if (sidebarFarmer) sidebarFarmer.textContent = "Guest";
      if (sidebarFarm) sidebarFarm.textContent = "Set up your farm";
      if (sidebarAvatar) sidebarAvatar.textContent = "+";
    }
  }

  function showSetupBanner() {
    const banner = document.getElementById("setup-farm-banner");
    if (!banner) return;
    banner.hidden = !!getFarmer();
  }

  global.HerdLineHistory = {
    getFarmer,
    saveFarmer,
    getCheckins,
    saveCheckin,
    getCheckin,
    deleteCheckin,
    formatDateLabel,
    saveFromPayloadElement,
    initDashboard,
    initCheckinsList,
    initCheckinDetail,
    updateNavAccount,
    showSetupBanner,
  };

  document.addEventListener("DOMContentLoaded", () => {
    updateNavAccount();
    showSetupBanner();
    saveFromPayloadElement();
    initDashboard();
    initCheckinsList();
    initCheckinDetail();
  });
})(window);
