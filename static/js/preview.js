document.addEventListener("DOMContentLoaded", function () {

    const fileInput = document.getElementById("imageInput");
    const preview = document.getElementById("previewImage");
    const previewContainer = document.getElementById("previewContainer");

    fileInput.addEventListener("change", function () {

        const file = this.files[0];

        if (!file) {

            previewContainer.style.display = "none";
            return;
        }

        preview.src = URL.createObjectURL(file);

        previewContainer.style.display = "block";

    });

});