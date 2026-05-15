const samples = [
  "agrivision_train_0000",
  "agrivision_train_0005",
  "agrivision_train_0010",
];

const lrImage = document.querySelector("#lr-image");
const bicubicImage = document.querySelector("#bicubic-image");
const hrImage = document.querySelector("#hr-image");
const tabs = document.querySelectorAll(".sample-tab");

function setSample(index) {
  const id = samples[index];
  lrImage.src = `assets/${id}_lr.png`;
  bicubicImage.src = `assets/${id}_bicubic.png`;
  hrImage.src = `assets/${id}_hr.png`;
  tabs.forEach((tab) => tab.classList.toggle("active", Number(tab.dataset.sample) === index));
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => setSample(Number(tab.dataset.sample)));
});
