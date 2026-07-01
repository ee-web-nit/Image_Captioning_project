document.addEventListener("DOMContentLoaded", () => {

    const form = document.getElementById("uploadForm");
    const spinner = document.getElementById("loadingSpinner");

    form.addEventListener("submit", () => {

        spinner.style.display = "block";

    });

});