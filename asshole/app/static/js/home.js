// Configuration
const config = {
  socketUrl: 'http://' + document.domain + ':' + location.port
};

// WebSocket Service
class SocketService {
  constructor(url) {
    this.socket = io.connect(url);
    this.setupListeners();
  }

  setupListeners() {
    this.socket.on('connect', () => {
      console.log('WebSocket connection established');
    });
  }

  emit(event, data) {
    this.socket.emit(event, data);
  }

  on(event, callback) {
    this.socket.on(event, callback);
  }
}

// API Service
class APIService {
  static async fetchJSON(url, options = {}) {
    try {
      const response = await fetch(url, options);
      const data = await response.json();
      return data;
    } catch (error) {
      console.error(`API Error (${url}):`, error);
      throw error;
    }
  }

  static async getUserInfo() {
    return this.fetchJSON('/get_user_info');
  }

  static async authenticate(action, username, password) {
    return this.fetchJSON('/' + action, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username, password })
    });
  }

  static async logout() {
    return this.fetchJSON('/logout', { method: 'POST' });
  }
}

// User Auth Controller
class AuthController {
  constructor(socketService) {
    this.socketService = socketService;
  }

  async initialize() {
    this.setupAuthForm();
    await this.updateUserStatus();
  }

  async updateUserStatus() {
    const data = await APIService.getUserInfo();
    const userStatusEl = document.getElementById("user-status");
    const newGameButton = document.getElementById("new_game_button");
    const loginForm = document.getElementById("login-form");
    const loginMessage = document.getElementById("login-message");
    const logoutBtn = document.getElementById("logout-btn");

    if (data.logged_in) {
      console.log("User is logged in");
      userStatusEl.innerText = "Logged in as " + data.username;
      newGameButton.style.display = "block";
      loginForm.style.display = "none";
      loginMessage.style.display = "none";
      logoutBtn.style.display = "inline";
    } else {
      console.log("User is NOT logged in");
      userStatusEl.innerText = "Not Logged In";
      loginForm.style.display = "block";
      newGameButton.style.display = "none";
      loginMessage.style.display = "block";
      logoutBtn.style.display = "none";
    }
    return data.logged_in;
  }

  setupAuthForm() {
    const authForm = document.getElementById('auth-form');
    if (!authForm) {
      console.log('auth_form not found!');
      return;
    }

    authForm.addEventListener("click", async (event) => {
      if (event.target.tagName !== "BUTTON") return;

      event.preventDefault();
      const action = event.target.getAttribute("data-action");
      const username = document.getElementById("username").value;
      const password = document.getElementById("password").value;

      if (!username || !password) {
        alert("Please enter a username and password.");
        return;
      }

      try {
        const data = await APIService.authenticate(action, username, password);
        if (data.success) {
          location.reload();
        } else {
          alert(`${this.capitalizeFirst(action)} failed: ${data.error}`);
        }
      } catch (error) {
        alert("An error occurred during authentication. Please try again.");
      }
    });

    // Setup logout button
    const logoutBtn = document.getElementById("logout-btn");
    if (logoutBtn) {
      logoutBtn.addEventListener("click", async () => {
        console.log("Requesting logout");
        this.socketService.emit('logout');
        await APIService.logout();
        window.location.href = '/';
      });
    }
  }

  capitalizeFirst(string) {
    return string.charAt(0).toUpperCase() + string.slice(1);
  }
}

// Game List Controller
class GameListController {
  constructor(socketService) {
    this.socketService = socketService;
  }

  initialize() {
    this.setupSocketListeners();
    this.setupNewGameButton();
  }

  setupSocketListeners() {
    this.socketService.on('update_game_list', (data) => {
      this.renderGameList(data.games);
    });

    this.socketService.on("joined_game", (data) => {
      window.location.href = "/game/" + data.game_id;
    });

    this.socketService.on('notify_player_joined', (data) => {
      console.log("Joined game: " + data.game_id);
      window.location.href = "/game/" + data.game_id;
    });
  }

  setupNewGameButton() {
    const newGameButton = document.getElementById('new_game_button');
    if (newGameButton) {
      newGameButton.addEventListener("click", () => {
        this.socketService.emit("new_game");
        this.refreshGameList();
      });
    }
  }

  refreshGameList() {
    console.log("Requesting game list refresh");
    this.socketService.emit('refresh_games');
  }

  renderGameList(games) {
    const gameList = document.getElementById("game-list");
    console.log("Updating game list");
    gameList.innerHTML = "";

    games.forEach((game) => {
      const li = document.createElement("li");
      li.innerHTML = `Game ${game.id} - ${game.players.length}/4 Players `;

      if (game.players.length <= 4) {
        const joinButton = document.createElement("button");

        if (game.players.length == 4) {
          joinButton.innerText = "View Game";
          joinButton.onclick = () => {
            window.location.href = "/view_game/" + game.id;
          };
        } else {
          joinButton.innerText = "Join Game";
          joinButton.onclick = () => {
            this.socketService.emit("join_game", { game_id: game.id });
          };
        }

        li.appendChild(joinButton);
      }

      gameList.appendChild(li);
    });
  }
}

// Application Controller
class HomeApp {
  constructor() {
    this.socketService = new SocketService(config.socketUrl);
    this.authController = new AuthController(this.socketService);
    this.gameListController = new GameListController(this.socketService);
  }

  async initialize() {
    await this.authController.initialize();
    this.gameListController.initialize();
    this.gameListController.refreshGameList();
  }
}

// Initialize the application when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
  const app = new HomeApp();
  app.initialize();
});
