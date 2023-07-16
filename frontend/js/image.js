const form = document.getElementById("myForm");
form.addEventListener("submit", function (event) {
  event.preventDefault(); // Formu normal gönderimden engelle

  const formData = new FormData(form);

  const url = "http://10.30.1.205:4040/digits"; // Kullanacağınız API endpoint URL'sini buraya değiştirin
  const options = {
    method: "POST",
    body: formData,
  };

  fetch(url, options)
    .then((response) => {
      if (response.ok) {
        return response.json();
      }
      return Promise.reject(response);
    })
    .then((data) => {
      // Yanıt verisini işleyin
      console.log(data);
      alert("Prediction: " + data['prediction']);
    })
    .catch((error) => {
      // Hataları işleyin
      console.error("Error:", error);
    });
});
