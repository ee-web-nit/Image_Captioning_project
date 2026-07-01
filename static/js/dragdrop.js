document.addEventListener("DOMContentLoaded", () => {

    const dropArea = document.getElementById("dropArea");
    const fileInput = document.getElementById("imageInput");

    ["dragenter", "dragover"].forEach(eventName => {

        dropArea.addEventListener(eventName, (e) => {

            e.preventDefault();
            e.stopPropagation();

            dropArea.classList.add("dragover");

        });

    });

    ["dragleave", "drop"].forEach(eventName => {

        dropArea.addEventListener(eventName, (e) => {

            e.preventDefault();
            e.stopPropagation();

            dropArea.classList.remove("dragover");

        });

    });

    dropArea.addEventListener("drop", (e) => {

        const files = e.dataTransfer.files;

        if (files.length > 0) {

            fileInput.files = files;

            fileInput.dispatchEvent(
                new Event("change", { bubbles: true })
            );

        }

    });
    const browseButton = document.getElementById("browseButton");

    browseButton.addEventListener("click", () => {

        fileInput.click();

    });
});