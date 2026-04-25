const CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary'];

const LABELS = {
  glioma:     'Glioma',
  meningioma: 'Meningioma',
  notumor:    'No Tumor',
  pituitary:  'Pituitary Tumor'
};

// ImageNet normalisation values the model was trained with
const MEAN = [0.485, 0.456, 0.406];
const STD  = [0.229, 0.224, 0.225];
const IMG_SIZE = 256;

let session = null;
let currentFile = null;

const pip        = document.getElementById('pip');
const statusText = document.getElementById('statusText');
const drop       = document.getElementById('drop');
const fileInput  = document.getElementById('fileInput');
const panel      = document.getElementById('panel');
const thumb      = document.getElementById('thumb');
const scanName   = document.getElementById('scanName');
const scanMeta   = document.getElementById('scanMeta');
const runBtn     = document.getElementById('runBtn');
const clearBtn   = document.getElementById('clearBtn');
const loader     = document.getElementById('loader');
const results    = document.getElementById('results');

// ── model loading ────────────────────────────────────────────────────────────

async function loadModel() {
  try {
    session = await ort.InferenceSession.create('./brain_tumor_model.onnx');
    pip.classList.add('ok');
    statusText.textContent = 'Model ready';
    if (currentFile) runBtn.disabled = false;
  } catch (e) {
    pip.classList.add('err');
    statusText.textContent = 'Model not found in folder';
    console.error(e);
  }
}

loadModel();

// ── file handling ────────────────────────────────────────────────────────────

function revokeThumb() {
  if (thumb.src) {
    URL.revokeObjectURL(thumb.src);
    thumb.src = '';
  }
}

function handleFile(f) {
  if (!f || !f.type.startsWith('image/')) return;

  revokeThumb();

  currentFile = f;
  thumb.src = URL.createObjectURL(f);
  scanName.textContent = f.name;
  scanMeta.textContent = (f.size / 1024).toFixed(1) + ' KB';

  drop.classList.add('hidden');
  panel.classList.add('on');
  results.classList.remove('on');

  if (session) runBtn.disabled = false;
}

drop.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', e => {
  if (e.target.files[0]) handleFile(e.target.files[0]);
});

drop.addEventListener('dragover', e => {
  e.preventDefault();
  drop.classList.add('over');
});

drop.addEventListener('dragleave', () => drop.classList.remove('over'));

drop.addEventListener('drop', e => {
  e.preventDefault();
  drop.classList.remove('over');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

clearBtn.addEventListener('click', () => {
  revokeThumb();
  currentFile = null;
  fileInput.value = '';

  panel.classList.remove('on');
  results.classList.remove('on');
  loader.classList.remove('on');
  drop.classList.remove('hidden');
});

// ── inference ────────────────────────────────────────────────────────────────

function preprocess(img) {
  const canvas = document.createElement('canvas');
  canvas.width  = IMG_SIZE;
  canvas.height = IMG_SIZE;

  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, IMG_SIZE, IMG_SIZE);

  const { data } = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE);
  const f32 = new Float32Array(3 * IMG_SIZE * IMG_SIZE);

  for (let y = 0; y < IMG_SIZE; y++) {
    for (let x = 0; x < IMG_SIZE; x++) {
      const px = (y * IMG_SIZE + x) * 4;
      for (let ch = 0; ch < 3; ch++) {
        f32[ch * IMG_SIZE * IMG_SIZE + y * IMG_SIZE + x] =
          (data[px + ch] / 255 - MEAN[ch]) / STD[ch];
      }
    }
  }

  return new ort.Tensor('float32', f32, [1, 3, IMG_SIZE, IMG_SIZE]);
}

function softmax(arr) {
  const max  = Math.max(...arr);
  const exps = arr.map(v => Math.exp(v - max));
  const sum  = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

runBtn.addEventListener('click', async () => {
  if (!session || !currentFile) return;

  runBtn.disabled = true;
  results.classList.remove('on');
  loader.classList.add('on');

  try {
    await thumb.decode();

    const tensor = preprocess(thumb);
    const output = await session.run({ [session.inputNames[0]]: tensor });
    const probs  = softmax(Array.from(output[session.outputNames[0]].data));
    const best   = probs.indexOf(Math.max(...probs));

    render(CLASSES[best], probs);
  } catch (err) {
    console.error(err);
    alert('Error during inference. Check the console for details.');
  } finally {
    loader.classList.remove('on');
    runBtn.disabled = false;
  }
});

// ── render results ───────────────────────────────────────────────────────────

function render(predClass, probs) {
  const isTumor = predClass !== 'notumor';

  const tag = document.getElementById('tag');
  tag.className   = 'tag ' + (isTumor ? 'tumor' : 'healthy');
  tag.textContent = isTumor ? 'Tumor detected' : 'No tumor';

  document.getElementById('verdictLabel').textContent = LABELS[predClass];

  const sorted = CLASSES
    .map((c, i) => ({ c, p: probs[i] }))
    .sort((a, b) => b.p - a.p);

  document.getElementById('barList').innerHTML = sorted.map(({ c, p }, i) => `
    <div class="bar-item">
      <span class="bar-label">${LABELS[c]}</span>
      <div class="bar-track">
        <div class="bar-fill ${i === 0 ? 'top' : ''}" data-w="${p * 100}"></div>
      </div>
      <span class="bar-pct">${(p * 100).toFixed(1)}%</span>
    </div>
  `).join('');

  results.classList.add('on');

  // trigger CSS transition on next frame
  requestAnimationFrame(() => {
    document.querySelectorAll('.bar-fill').forEach(el => {
      el.style.width = el.dataset.w + '%';
    });
  });
}
