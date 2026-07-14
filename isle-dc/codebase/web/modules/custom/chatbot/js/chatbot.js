(function (Drupal) {

  Drupal.behaviors.chatbot = {
    attach: function (context, settings) {

      const button = document.querySelector("#send-btn");
      const input = document.querySelector("#user-message");
      const chatWindow = document.querySelector("#chat-window");

      if (!button || button.dataset.loaded) {
        return;
      }

      button.dataset.loaded = true;

      button.addEventListener("click", function () {

        const message = input.value.trim();

        if (message === "") {
          return;
        }

        // User message
        const userDiv = document.createElement("div");
        userDiv.className = "user-message";
        userDiv.innerText = message;
        chatWindow.appendChild(userDiv);

        // Temporary bot reply
        const botDiv = document.createElement("div");
        botDiv.className = "bot-message";
        botDiv.innerText = "Thinking...";
        chatWindow.appendChild(botDiv);

        chatWindow.scrollTop = chatWindow.scrollHeight;

        input.value = "";
        fetch("/chatbot/message", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                message: message
            })
        })
        .then(response => response.json())
        .then(data => {
            botDiv.innerText = data.reply;
        })
        .catch(error => {
            console.error(error);
            botDiv.innerText = "Something went wrong.";
        });

      });

    }
  };

})(Drupal);