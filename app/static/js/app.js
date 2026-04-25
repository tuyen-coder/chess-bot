const PIECES = {
  wK: "♔",
  wQ: "♕",
  wR: "♖",
  wB: "♗",
  wN: "♘",
  wp: "♙",
  bK: "♚",
  bQ: "♛",
  bR: "♜",
  bB: "♝",
  bN: "♞",
  bp: "♟",
};

const MODES = [
  ["Online Match", "online"],
  ["1. Player vs Player", "pvp"],
  ["2. Player vs Random AI", "pvr"],
  ["3. Player vs Minimax AI", "pvm"],
  ["4. Random vs Minimax", "rvm"],
  ["5. Minimax vs Minimax", "mvm"],
  ["6. Player vs ML AI", "pvl"],
  ["7. ML vs Random", "mlr"],
];

const OPPONENT_LABELS = {
  random: "Random AI",
  minimax: "Minimax AI",
  ml: "ML AI",
};

const app = document.getElementById("app");
let state = null;
let currentView = "auth";
let aiLoopRunning = false;
let onlineLoopRunning = false;
let flash = null;

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    method: options.method || "GET",
    headers: { "Content-Type": "application/json" },
    body: options.body ? JSON.stringify(options.body) : undefined,
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || `Request failed: ${response.status}`);
  }
  return data;
}

function setFlash(message, type = "info") {
  flash = { message, type };
  render();
}

function clearFlash() {
  flash = null;
}

function syncViewFromState() {
  if (!state?.user) {
    currentView = "auth";
    return;
  }
  if (state.online) {
    currentView = "online";
    return;
  }
  if (!["menu", "game", "profile"].includes(currentView)) {
    currentView = state.inMenu ? "menu" : "game";
  }
  if (state.user && currentView === "auth") {
    currentView = state.inMenu ? "menu" : "game";
  }
}

function renderFlash() {
  if (!flash) return "";
  return `<section class="flash-card ${flash.type === "error" ? "flash-error" : ""}">${escapeHtml(flash.message)}</section>`;
}

function renderTopNav() {
  if (!state?.user) return "";
  return `
    <header class="top-nav">
      <div>
        <h1>Chess Studio</h1>
        <p>Signed in as <strong>${escapeHtml(state.user.username)}</strong></p>
      </div>
      <div class="top-nav-actions">
        <button class="secondary-button" data-nav="menu">Menu</button>
        <button class="secondary-button" data-nav="profile">Profile</button>
        <button class="secondary-button" id="logout-button">Log out</button>
      </div>
    </header>
  `;
}

function statsRows() {
  if (!state?.stats) return "";
  return Object.entries(state.stats).map(([key, value]) => `
    <tr>
      <td>${OPPONENT_LABELS[key]}</td>
      <td>${value.wins}</td>
      <td>${value.losses}</td>
      <td>${value.draws}</td>
      <td>${value.winRate}%</td>
    </tr>
  `).join("");
}

function leaderboardRows() {
  const rows = state?.leaderboard || [];
  if (!rows.length) {
    return `<tr><td colspan="6">No rated games yet.</td></tr>`;
  }
  return rows.map((player) => `
    <tr>
      <td>${player.rank}</td>
      <td>${escapeHtml(player.username)}</td>
      <td>${player.elo}</td>
      <td>${player.wins}</td>
      <td>${player.losses}</td>
      <td>${player.draws}</td>
    </tr>
  `).join("");
}

function renderLeaderboardTable() {
  return `
    <table class="stats-table">
      <thead>
        <tr>
          <th>Rank</th>
          <th>Player</th>
          <th>Elo</th>
          <th>Wins</th>
          <th>Losses</th>
          <th>Draws</th>
        </tr>
      </thead>
      <tbody>${leaderboardRows()}</tbody>
    </table>
  `;
}

function renderAuthPage() {
  app.innerHTML = `
    <main class="page page-narrow">
      <section class="hero">
        <h1>Chess Studio</h1>
        <p>Create an account or log in to start playing and store your stats in the backend SQL database.</p>
      </section>
      ${renderFlash()}
      <section class="auth-grid">
        <section class="card auth-card">
          <h2>Create Account</h2>
          <form id="register-form" class="auth-form">
            <label>
              Username
              <input name="username" type="text" required minlength="3" maxlength="24" pattern="[A-Za-z0-9_-]{3,24}" />
            </label>
            <label>
              Password
              <input name="password" type="password" required minlength="8" maxlength="128" />
            </label>
            <button class="primary-button" type="submit">Register</button>
          </form>
        </section>
        <section class="card auth-card">
          <h2>Log In</h2>
          <form id="login-form" class="auth-form">
            <label>
              Username
              <input name="username" type="text" required maxlength="24" pattern="[A-Za-z0-9_-]{3,24}" />
            </label>
            <label>
              Password
              <input name="password" type="password" required minlength="8" maxlength="128" />
            </label>
            <button class="primary-button" type="submit">Log In</button>
          </form>
        </section>
      </section>
    </main>
  `;

  const registerForm = document.getElementById("register-form");
  registerForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(registerForm);
    try {
      state = await api("/api/register", {
        method: "POST",
        body: {
          username: formData.get("username"),
          password: formData.get("password"),
        },
      });
      currentView = "menu";
      setFlash("Account created and logged in.", "success");
    } catch (error) {
      setFlash(error.message, "error");
    }
  });

  const loginForm = document.getElementById("login-form");
  loginForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(loginForm);
    try {
      state = await api("/api/login", {
        method: "POST",
        body: {
          username: formData.get("username"),
          password: formData.get("password"),
        },
      });
      currentView = "menu";
      setFlash("Logged in successfully.", "success");
    } catch (error) {
      setFlash(error.message, "error");
    }
  });
}

function modeCard([label, mode]) {
  return `
    <section class="mode-card">
      <h2>${label}</h2>
      <button class="primary-button" data-mode="${mode}">Start</button>
    </section>
  `;
}

function renderMenuPage() {
  app.innerHTML = `
    <main class="page">
      ${renderTopNav()}
      ${renderFlash()}
      <section class="hero">
        <h1>Welcome back, ${escapeHtml(state.user.username)}</h1>
        <p>Pick a mode to start a match, then visit your profile page to review stored stats or update your account details.</p>
      </section>
      <section class="card quick-stats">
        <h2>Quick Stats</h2>
        <table class="stats-table">
          <thead>
            <tr>
              <th>Opponent</th>
              <th>Wins</th>
              <th>Losses</th>
              <th>Draws</th>
              <th>Win rate</th>
            </tr>
          </thead>
          <tbody>${statsRows()}</tbody>
        </table>
      </section>
      <section class="card quick-stats">
        <h2>Leaderboard</h2>
        ${renderLeaderboardTable()}
      </section>
      <section class="modes-grid">
        ${MODES.map(modeCard).join("")}
      </section>
    </main>
  `;

  bindAuthenticatedNav();
  app.querySelectorAll("[data-mode]").forEach((button) => {
    button.addEventListener("click", async () => {
      try {
        if (button.dataset.mode === "online") {
          state = await api("/api/matchmaking/join", { method: "POST" });
          currentView = "online";
          clearFlash();
          render();
          runOnlineLoopIfNeeded();
          return;
        }
        state = await api("/api/start", {
          method: "POST",
          body: { mode: button.dataset.mode },
        });
        currentView = "game";
        clearFlash();
        render();
        runAiLoopIfNeeded();
      } catch (error) {
        setFlash(error.message, "error");
      }
    });
  });
}

function buildBoardGrid() {
  const topCoords = "abcdefgh".split("").map((file, index) =>
    `<div class="coord" style="grid-row:1;grid-column:${index + 2};">${file}</div>`
  ).join("");

  const bottomCoords = "abcdefgh".split("").map((file, index) =>
    `<div class="coord" style="grid-row:10;grid-column:${index + 2};">${file}</div>`
  ).join("");

  const rankLabels = Array.from({ length: 8 }, (_, index) => 8 - index).map((rank, index) => `
      <div class="coord" style="grid-row:${index + 2};grid-column:1;">${rank}</div>
      <div class="coord" style="grid-row:${index + 2};grid-column:10;">${rank}</div>
    `
  ).join("");

  const selectedKey = state.selected ? `${state.selected[0]}-${state.selected[1]}` : null;
  const legalKeys = new Set((state.legalMoves || []).map(([row, col]) => `${row}-${col}`));
  const checkKey = state.checkSquare ? `${state.checkSquare[0]}-${state.checkSquare[1]}` : null;

  const squares = state.board.flatMap((row, rowIndex) =>
    row.map((piece, colIndex) => {
      const key = `${rowIndex}-${colIndex}`;
      const classes = ["square", (rowIndex + colIndex) % 2 === 0 ? "light" : "dark"];
      if (selectedKey === key) classes.push("selected");
      if (legalKeys.has(key)) classes.push("legal");
      if (checkKey === key) classes.push("check");

      return `
        <div
          class="${classes.join(" ")}"
          style="grid-row:${rowIndex + 2};grid-column:${colIndex + 2};"
          data-row="${rowIndex}"
          data-col="${colIndex}"
        >${piece === "--" ? "" : PIECES[piece]}</div>
      `;
    })
  ).join("");

  return `
    <div class="board-wrap ${state.canInteract ? "" : "loading"}">
      ${topCoords}
      ${bottomCoords}
      ${rankLabels}
      ${squares}
    </div>
  `;
}

function renderGamePage() {
  const moveLogText = state.moveLog.length
    ? state.moveLog.reduce((lines, move, index, allMoves) => {
        if (index % 2 === 0) {
          const turn = Math.floor(index / 2) + 1;
          const blackMove = allMoves[index + 1] || "";
          lines.push(`${turn}. ${move}${blackMove ? `   ${blackMove}` : ""}`);
        }
        return lines;
      }, []).join("\n")
    : "The move log will appear here once the game begins.";

  app.innerHTML = `
    <main class="page">
      ${renderTopNav()}
      ${renderFlash()}
      <section class="game-layout">
        <section class="board-panel">
          <div class="board-header">
            <div>
              <h1>Match Board</h1>
              <p class="muted">${escapeHtml(state.modeLabel)}${state.playerColor ? ` - You are ${escapeHtml(state.playerColor)}` : ""}</p>
            </div>
            <button class="secondary-button" id="back-button">Back to Menu</button>
          </div>
          ${buildBoardGrid()}
          <div class="info-bar">
            <div class="info-line">
              <span class="label-strong">Selected:</span>
              <span>${escapeHtml(state.selectionDisplay)}</span>
              <span class="label-strong">Legal moves:</span>
              <span>${escapeHtml(state.selection.legalMovesText)}</span>
            </div>
          </div>
        </section>

        <aside class="sidebar">
          <section class="card">
            <h2>Overview</h2>
            <div class="info-row"><span class="muted">Mode</span><span>${escapeHtml(state.modeLabel)}</span></div>
            ${state.opponent ? `<div class="info-row"><span class="muted">Opponent</span><span>${escapeHtml(state.opponent.username)}</span></div>` : ""}
            ${state.playerColor ? `<div class="info-row"><span class="muted">Color</span><span>${escapeHtml(state.playerColor)}</span></div>` : ""}
            <div class="info-row"><span class="muted">Turn</span><span>${escapeHtml(state.turn)}</span></div>
            <div class="info-row"><span class="muted">Status</span><span class="status-pill">${escapeHtml(state.status)}</span></div>
            <div class="info-row"><span class="muted">Last move</span><span>${escapeHtml(state.lastMove)}</span></div>
            <div class="info-row"><span class="muted">Move count</span><span>${state.moveCount}</span></div>
          </section>

          <section class="card">
            <h2>Selected Piece</h2>
            <div class="info-row"><span class="muted">Square</span><span>${escapeHtml(state.selectionDisplay)}</span></div>
            <div class="info-row"><span class="muted">Legal moves</span><span>${escapeHtml(state.selection.legalMovesText)}</span></div>
          </section>

          <section class="card">
            <h2>Account</h2>
            <div class="info-row"><span class="muted">Player</span><span>${escapeHtml(state.user.username)}</span></div>
            <div class="info-row"><span class="muted">Elo</span><span>${state.user.elo}</span></div>
            <div class="info-row"><span class="muted">Profile</span><span>View stats and update username/password from the profile page.</span></div>
          </section>

          <section class="card">
            <h2>Move Log</h2>
            <div class="move-log">${escapeHtml(moveLogText)}</div>
          </section>
        </aside>
      </section>
    </main>
  `;

  bindAuthenticatedNav();

  document.getElementById("back-button").addEventListener("click", async () => {
    state = await api(state.online ? "/api/matchmaking/leave" : "/api/menu", { method: "POST" });
    currentView = "menu";
    clearFlash();
    render();
  });

  app.querySelectorAll(".square").forEach((square) => {
    square.addEventListener("click", async () => {
      if (!state.canInteract) return;
      try {
        state = await api(state.online ? "/api/matchmaking/click" : "/api/click", {
          method: "POST",
          body: {
            row: Number(square.dataset.row),
            col: Number(square.dataset.col),
          },
        });
        clearFlash();
        render();
        if (state.online) {
          runOnlineLoopIfNeeded();
        } else {
          runAiLoopIfNeeded();
        }
      } catch (error) {
        setFlash(error.message, "error");
      }
    });
  });
}

function renderWaitingPage() {
  app.innerHTML = `
    <main class="page page-medium">
      ${renderTopNav()}
      ${renderFlash()}
      <section class="hero">
        <h1>Online Match</h1>
        <p>${escapeHtml(state.status || "Waiting for another player")}</p>
      </section>
      <section class="card">
        <h2>Queue</h2>
        <div class="info-row"><span class="muted">Player</span><span>${escapeHtml(state.user.username)}</span></div>
        <div class="info-row"><span class="muted">Elo</span><span>${state.user.elo}</span></div>
        <button class="secondary-button" id="leave-queue-button">Leave Queue</button>
      </section>
      <section class="card leaderboard-card">
        <h2>Leaderboard</h2>
        ${renderLeaderboardTable()}
      </section>
    </main>
  `;

  bindAuthenticatedNav();
  document.getElementById("leave-queue-button").addEventListener("click", async () => {
    state = await api("/api/matchmaking/leave", { method: "POST" });
    currentView = "menu";
    clearFlash();
    render();
  });
}

function renderProfilePage() {
  app.innerHTML = `
    <main class="page page-medium">
      ${renderTopNav()}
      ${renderFlash()}
      <section class="hero">
        <h1>Profile</h1>
        <p>Review your stored stats and update your account details.</p>
      </section>

      <section class="profile-layout">
        <section class="card">
          <h2>Account Details</h2>
          <div class="info-row"><span class="muted">Username</span><span>${escapeHtml(state.user.username)}</span></div>
          <div class="info-row"><span class="muted">Elo</span><span>${state.user.elo}</span></div>
          <div class="info-row"><span class="muted">Online</span><span>${state.user.multiplayer_wins}W ${state.user.multiplayer_losses}L ${state.user.multiplayer_draws}D</span></div>
          <div class="info-row"><span class="muted">Created</span><span>${escapeHtml(state.user.created_at || "Unknown")}</span></div>
        </section>

        <section class="card">
          <h2>Leaderboard</h2>
          ${renderLeaderboardTable()}
        </section>

        <section class="card">
          <h2>Performance Stats</h2>
          <table class="stats-table">
            <thead>
              <tr>
                <th>Opponent</th>
                <th>Wins</th>
                <th>Losses</th>
                <th>Draws</th>
                <th>Win rate</th>
              </tr>
            </thead>
            <tbody>${statsRows()}</tbody>
          </table>
        </section>

        <section class="card">
          <h2>Change Username</h2>
          <form id="username-form" class="auth-form">
            <label>
              New username
              <input name="username" type="text" required minlength="3" maxlength="24" pattern="[A-Za-z0-9_-]{3,24}" value="${escapeHtml(state.user.username)}" />
            </label>
            <button class="primary-button" type="submit">Update Username</button>
          </form>
        </section>

        <section class="card">
          <h2>Change Password</h2>
          <form id="password-form" class="auth-form">
            <label>
              Current password
              <input name="currentPassword" type="password" required minlength="8" maxlength="128" />
            </label>
            <label>
              New password
              <input name="newPassword" type="password" required minlength="8" maxlength="128" />
            </label>
            <button class="primary-button" type="submit">Update Password</button>
          </form>
        </section>
      </section>
    </main>
  `;

  bindAuthenticatedNav();

  document.getElementById("username-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    try {
      state = await api("/api/profile/username", {
        method: "POST",
        body: { username: formData.get("username") },
      });
      setFlash("Username updated.", "success");
    } catch (error) {
      setFlash(error.message, "error");
    }
  });

  document.getElementById("password-form").addEventListener("submit", async (event) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    try {
      state = await api("/api/profile/password", {
        method: "POST",
        body: {
          currentPassword: formData.get("currentPassword"),
          newPassword: formData.get("newPassword"),
        },
      });
      event.currentTarget.reset();
      setFlash("Password updated.", "success");
    } catch (error) {
      setFlash(error.message, "error");
    }
  });
}

function bindAuthenticatedNav() {
  app.querySelectorAll("[data-nav]").forEach((button) => {
    button.addEventListener("click", async () => {
      currentView = button.dataset.nav;
      if (currentView === "menu") {
        state = await api("/api/menu", { method: "POST" });
      }
      clearFlash();
      render();
    });
  });

  const logoutButton = document.getElementById("logout-button");
  if (logoutButton) {
    logoutButton.addEventListener("click", async () => {
      state = await api("/api/logout", { method: "POST" });
      currentView = "auth";
      setFlash("Logged out.", "success");
    });
  }
}

function render() {
  syncViewFromState();
  if (currentView === "auth") {
    renderAuthPage();
    return;
  }
  if (currentView === "profile") {
    renderProfilePage();
    return;
  }
  if (currentView === "online") {
    if (state.waitingForOpponent) {
      renderWaitingPage();
    } else {
      renderGamePage();
    }
    return;
  }
  if (currentView === "game" && !state.inMenu) {
    renderGamePage();
    return;
  }
  renderMenuPage();
}

async function runAiLoopIfNeeded() {
  if (aiLoopRunning || !state || !state.aiWaiting || state.gameOver) {
    return;
  }

  aiLoopRunning = true;

  try {
    while (state && state.aiWaiting && !state.gameOver) {
      await new Promise((resolve) => setTimeout(resolve, state.aiDelayMs || 800));
      state = await api("/api/ai-step", { method: "POST" });
      if (state.gameOver) {
        setFlash(state.gameResult || "Game finished.", "success");
      } else {
        render();
      }
    }
  } catch (error) {
    setFlash(error.message, "error");
  } finally {
    aiLoopRunning = false;
  }
}

async function runOnlineLoopIfNeeded() {
  if (onlineLoopRunning || !state || !state.online || state.gameOver) {
    return;
  }

  onlineLoopRunning = true;

  try {
    while (state && state.online && !state.gameOver) {
      await new Promise((resolve) => setTimeout(resolve, 1200));
      state = await api("/api/matchmaking/state", { method: "POST" });
      render();
      if (!state.online || state.gameOver) {
        break;
      }
    }
    if (state?.gameOver) {
      setFlash(state.gameResult || "Game finished.", "success");
    }
  } catch (error) {
    setFlash(error.message, "error");
  } finally {
    onlineLoopRunning = false;
  }
}

async function bootstrap() {
  state = await api("/api/state");
  syncViewFromState();
  render();
  if (state.online) {
    runOnlineLoopIfNeeded();
  } else {
    runAiLoopIfNeeded();
  }
}

bootstrap().catch((error) => {
  console.error(error);
  app.innerHTML = `<main class="page"><section class="card"><h2>Web UI failed to load</h2><p>${escapeHtml(error.message)}</p></section></main>`;
});
