var socket = io.connect('http://' + document.domain + ':' + location.port);

socket.on('connect', function() {
    console.log('WebSocket connection established');
});

document.addEventListener("DOMContentLoaded", function () {
    fetchUserInfo(); // Get user info on load and setup displayed fields
    fetchGames(); // Request game list from server
    setup_auth(); // Ensure authentication buttons are registered
});

socket.on('update_game_list', function (data) {
    var gameList = document.getElementById("game-list");
    console.log("Updating game list");
    gameList.innerHTML = ""; // Clear list

    data.games.forEach(function (game) {
        var li = document.createElement("li");
        li.innerHTML = `Game ${game.id} - ${game.players.length}/4 Players `;

        if (game.players.length <= 4) {
            var joinLink = document.createElement("a");
            if (game.players.length == 4) {
                joinLink.href = "/view_game/" + game.id;
                joinLink.innerText = "View Game";
            } else {
                joinLink.href = "/join_game/" + game.id;
                joinLink.innerText = "Join Game";
            }
            li.appendChild(joinLink);
        }

        gameList.appendChild(li);
    });
});

function log_out() {
    console.log("Requesting logout");
    socket.emit('logout');
    // Also make HTTP request to ensure session is cleared
    fetch('/logout', {method: 'POST'})
        .then(() => window.location.href = '/');
}

function fetchUserInfo() {
    console.log("Fetching user info");
    fetch('/get_user_info')
        .then(response => response.json())
        .then(data => {
            if (data.logged_in) {
                console.log("User is logged in");
                document.getElementById("user-status").innerText = "Logged in as " + data.username;
                document.getElementById("new_game_button").style.display = "block";
                document.getElementById('new_game_button').addEventListener("click", function () {
                    socket.emit("new_game");
                    fetchGames(); // Refresh the game list after requesting a new game
                });
                document.getElementById("login-message").style.display = "none";
                document.getElementById("login-form").style.display = "none";
                document.getElementById("logout-btn").style.display = "inline";
            } else {
                console.log("User is NOT logged in");
                document.getElementById("user-status").innerText = "Not Logged In";
                document.getElementById("login-form").style.display = "block";
                document.getElementById("new_game_button").style.display = "none";
                document.getElementById("login-message").style.display = "block";
            }
        })
        .catch(error => {
            console.error("Error fetching user info:", error);
        });
}

function setup_auth() {
    var auth_form = document.getElementById('auth-form');
    if (auth_form === null) {
        console.log('auth_form not found!');
        return;
    }

    document.getElementById("auth-form").addEventListener("click", function(event) {
        if (event.target.tagName !== "BUTTON") return; // Ignore non-button clicks

        event.preventDefault();
        let action = event.target.getAttribute("data-action");
        let username = document.getElementById("username").value;
        let password = document.getElementById("password").value;

        if (!username || !password) {
            alert("Please enter a username and password.");
            return;
        }

        console.log("Fetching action: " + action);

        fetch('/' + action, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username, password })
        }).then(response => response.json())
          .then(data => {
              if (data.success) {
                  location.reload(); // Reload to update UI
              } else {
                  alert(`${action.charAt(0).toUpperCase() + action.slice(1)} failed: ${data.error}`);
              }
          })
          .catch(error => {
              console.error("Authentication error:", error);
              alert("An error occurred during authentication. Please try again.");
          });
    });
}

function fetchGames() {
    console.log("Requesting game list refresh");
    socket.emit('refresh_games'); // Request the latest game list
}

socket.on('game_created', function(data) {
    console.log("New game created: " + data.game_id);
    window.location.href = "/game/" + data.game_id; // Redirect to game page
});