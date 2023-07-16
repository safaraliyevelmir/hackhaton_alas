const form = document.getElementById("myForm");
const pred = document.getElementById("pred");
form.addEventListener("submit", function (event) {
  event.preventDefault(); // Formu normal gönderimden engelle

  var serializeForm = function (form) {
    var obj = {};
    var formData = new FormData(form);
    for (var key of formData.keys()) {
      obj[key] = formData.get(key);
    }
    return obj;
  };
  const data = serializeForm(event.target);
  console.log(data);

  const url = "http://10.30.1.205:4040/predict"; // Kullanacağınız API endpoint URL'sini buraya değiştirin
  const options = {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  };

  console.log(data);

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
      pred.innerHTML = data["prediction"];
    })
    .catch((error) => {
      // Hataları işleyin
      console.error("Error:", error);
    });
});
