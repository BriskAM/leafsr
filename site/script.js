const samples = [
  { id: "agrivision_train_0000", bicubicMae: 22.21, leafsrMae: 21.37 },
  { id: "agrivision_train_0005", bicubicMae: 14.96, leafsrMae: 13.86 },
  { id: "agrivision_train_0010", bicubicMae: 20.21, leafsrMae: 18.85 },
];

const lrImage = document.querySelector("#lr-image");
const bicubicImage = document.querySelector("#bicubic-image");
const leafsrImage = document.querySelector("#leafsr-image");
const hrImage = document.querySelector("#hr-image");
const bicubicMae = document.querySelector("#bicubic-mae");
const leafsrMae = document.querySelector("#leafsr-mae");
const sampleGain = document.querySelector("#sample-gain");
const tabs = document.querySelectorAll(".sample-tab");

function setSample(index) {
  const sample = samples[index];
  const id = sample.id;
  lrImage.src = `assets/${id}_lr.png`;
  bicubicImage.src = `assets/${id}_bicubic.png`;
  leafsrImage.src = `assets/predictions/${id}.png`;
  hrImage.src = `assets/${id}_hr.png`;
  bicubicMae.textContent = sample.bicubicMae.toFixed(2);
  leafsrMae.textContent = sample.leafsrMae.toFixed(2);
  sampleGain.textContent = `${(((sample.bicubicMae - sample.leafsrMae) / sample.bicubicMae) * 100).toFixed(1)}%`;
  tabs.forEach((tab) => tab.classList.toggle("active", Number(tab.dataset.sample) === index));
}

tabs.forEach((tab) => {
  tab.addEventListener("click", () => setSample(Number(tab.dataset.sample)));
});
