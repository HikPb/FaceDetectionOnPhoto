const imageUpload = document.getElementById('imageUpload')
const div1 = document.getElementById('img-div');

Promise.all([
  faceapi.nets.faceRecognitionNet.loadFromUri('/models'),
  faceapi.nets.faceLandmark68Net.loadFromUri('/models'),
  faceapi.nets.ssdMobilenetv1.loadFromUri('/models')
]).then(start)

async function start() {
  const container = document.createElement('div')
  container.style.position = 'relative'
  div1.append(container)
  const labeledFaceDescriptors = await detectLabeledFace()
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.7)
  let image
  let canvas
  div1.append('Loaded')
  imageUpload.addEventListener('change', async () => {
    if (image) image.remove()
    if (canvas) canvas.remove()
    image = await faceapi.bufferToImage(imageUpload.files[0])
    container.append(image)
    canvas = faceapi.createCanvasFromMedia(image)
    container.append(canvas)
    const displaySize = { width: image.width, height: image.height }
    faceapi.matchDimensions(canvas, displaySize)
    const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
    const resizedDetections = faceapi.resizeResults(detections, displaySize)
    const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
    results.forEach((result, i) => {
      const box = resizedDetections[i].detection.box
      const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() })
      drawBox.draw(canvas)
    })
  })
}

async function detectLabeledFace() {
    const label = "Lisa";
    const numberImage = 14;
    const descriptions = [];
    for (let i = 1; i <= numberImage; i++) {
    const img = await faceapi.fetchImage( 
    `http://localhost:5500/data/Lisa/${i}.JPG`
    );
    const detection = await faceapi
    .detectSingleFace(img).withFaceLandmarks()
    .withFaceDescriptor(); 
    descriptions.push(detection.descriptor);
    }
    return new faceapi.LabeledFaceDescriptors(label, descriptions);
}