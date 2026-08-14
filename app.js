let session = null;
let coordMap = null;
let roleToColor = null;
let nameToGrade = null;
let boardImage = null;
let Guesser = null;

function loadBoardImage() {
  boardImage = new Image();
  boardImage.src = "kilterboard.jpeg";
  boardImage.onload = function() {
    drawClimb([]);   // draw the board immediately, with zero holds on top yet
  };
}f

async function init() {
  session = await ort.InferenceSession.create("KilterGen.onnx");
  Guesser = await ort.InferenceSession.create("guesser.onnx");

  const response = await fetch("KilterMap.json");
  const data = await response.json();
  coordMap = data.coord_map;
  roleToColor = data.role_to_color;  // get the maps to map string -> actual position/color
  nameToGrade = data.name_to_grade;

  loadBoardImage();

  document.getElementById("generate").addEventListener("click", onGenerate);
}

function buildCausalMask(len) { // define what we can talk to/ what we can see when we generate
  let arr = new Float32Array(len * len);
  for (let i = 0; i < len; i++) {   // we need to make the casual mask. here it should be that bottom triangle because
    for (let j = 0; j < len; j++) { // we want everything to see itself and what comes before it.
      if (j <= i) {
        arr[i* len + j] = 0;
      } else {
        arr[i * len + j] = -Infinity; 
      }
    }
  }
  return arr;
}

function stackChannels(channelsInOrder) {
  const size = 44 * 47;
  const out = new Float32Array(channelsInOrder.length * size);
  channelsInOrder.forEach((channel, i) => out.set(channel, i * size));
  return out;
}

async function generateClimb(angle, grade, nomatch) {
  angle = angle - (angle % 5);
  if (grade > 13) {
    grade = 13;
  }
  grade = nameToGrade["V" + grade];
  let token = [(angle / 5) + 1511, grade + 70 / 5 + 1511, nomatch + 70 / 5 + 1511 + 39];

  while (true) {
    const seqLen = token.length;
    const tokenTensor = new ort.Tensor("int64", BigInt64Array.from(token.map(BigInt)), [1, seqLen]);
    const attentionTensor = new ort.Tensor("float32", new Float32Array(seqLen).fill(1), [1, seqLen]);
    const casualTensor = new ort.Tensor("float32", buildCausalMask(seqLen), [seqLen, seqLen]);  // create input tensors
    const final = await session.run({ tokens: tokenTensor, attention: attentionTensor, casual: casualTensor })
    const logitsData = final.logits.data;
    const dims = final.logits.dims;
    const size = dims[2];
    const lastPosition = (seqLen - 1) * size;
    const lastLogits = logitsData.slice(lastPosition, lastPosition + size);

    // softmax like in python
    const maxVal = Math.max(...lastLogits);
    const exps = lastLogits.map(x => Math.exp(x - maxVal));
    const sumExps = exps.reduce((total, x) => total + x, 0);
    const probs = exps.map(x => x / sumExps);


    //Choose a random logit
    const r = Math.random();
    let cumulative = 0;
    let nextToken = probs.length - 1;
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (r < cumulative) {
        nextToken = i;
        break;
      }
    }

    token.push(nextToken);
    if (nextToken === 0 || token.length >= 150) {
      break;
    }
  }

  return token;
}

async function generateOneAttempt(angle, grade, nomatch, Guesser) {
  let holds = await generateClimb(angle, grade, nomatch, Guesser);
  holds.splice(0, 3);
  if (holds[holds.length - 1] === 0) {
    holds.pop();
  }
  let holdarr = [];
  for (let i = 0; i < holds.length - 1; i += 2) {
    const [x, y] = coordMap[holds[i]];
    const color = roleToColor[holds[i + 1]];
    const hold = [x, y, color];
    holdarr.push(hold)
  }
  return holdarr;
}

async function onGenerate() {
  const angle = parseInt(document.getElementById("angle").value);
  const grade = parseInt(document.getElementById("grade").value);
  const nomatch = parseInt(document.getElementById("Match").checked ? 1 : 0);

  let holdarr = [];
  let numberOfClimbs = 0;
  let bestProb = 0;
  while(numberOfClimbs <= 15) {
    const candidate = await generateOneAttempt(angle, grade, nomatch, Guesser);

    let foot = 0;
    let greenDistance = [];
    let purpleDistance = [];
    for (const hold of candidate) {
      if (hold[2] === "start") {
        greenDistance.push(hold[0]);
        greenDistance.push(hold[1]);
      }
      if (hold[2] === "finish") {
        purpleDistance.push(hold[0]);
        purpleDistance.push(hold[1]);
      }
      if (hold[2] === "foot") {
        foot++;
      }
    }

    if (greenDistance.length > 0 && greenDistance.length <= 4 && purpleDistance.length > 0 && purpleDistance.length <= 4 && foot > 0) {
      let reachable = true;
      if (greenDistance.length == 4) {
        let greenLengthy = greenDistance[3] - greenDistance[1];
        let greenLengthx = greenDistance[2] - greenDistance[0];
        if (Math.sqrt((greenLengthy * greenLengthy) + (greenLengthx * greenLengthx)) > 51) {
          reachable = false;
        }
      }
      if (purpleDistance.length == 4) {
        let purpleLengthy = purpleDistance[3] - purpleDistance[1];
        let purpleLengthx = purpleDistance[2] - purpleDistance[0];
        if (Math.sqrt((purpleLengthy * purpleLengthy) + (purpleLengthx * purpleLengthx)) > 51) {
          reachable = false;
        }
      }

      if (reachable) {
        const prob = await gradeClimb(candidate, nomatch, angle, grade, Guesser);
        if (prob > bestProb) {
          holdarr = candidate;
          bestProb = prob;
        }
        numberOfClimbs++;
      }
    }
  }
  drawClimb(holdarr);
  document.getElementById("confidence").textContent = "Confidence: " + (bestProb * 100).toFixed(1) + "% V" + grade;
}





async function gradeClimb(holds, noMatch, angle, grade, Guesser) {
  let feet = new Float32Array(44*47);
  let finish = new Float32Array(44*47);
  let hands = new Float32Array(44*47);
  let start = new Float32Array(44*47);


  for(let i = 0; i < holds.length; i++) {
    const row = Math.floor((holds[i][1] - 4) / 4);
    const col = Math.floor((holds[i][0] + 20) / 4);
    if (holds[i][2] == "foot") {
      feet[row * 47 + col] = 1;
    }
    if (holds[i][2] == "start") {
      start[row * 47 + col] = 1;
    }
    if (holds[i][2] == "finish") {
      finish[row * 47 + col] =1;

    }
    if (holds[i][2] == "middle") {
      hands[row * 47 + col] = 1;

    }
  }

  const grid = stackChannels([feet, hands, start, finish]);
  const gridTensor = new ort.Tensor("float32", grid, [1,4,44,47]);
  const auxTensor = new ort.Tensor("float32", new Float32Array([angle, noMatch]), [1,2]);
  const result = await Guesser.run({grid: gridTensor, aux : auxTensor})
  const logits = result.logits.data;

  // softmax like in generateClimb
  const maxVal = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - maxVal));
  const sumExps = exps.reduce((total, x) => total + x, 0);
  const prob = exps.map(x => x / sumExps);
  

  let finalProb = 0;
  if (grade == 0) {
    finalProb = prob[0] + prob[1] + prob[2];
  }
  if (grade == 1) {
    finalProb = prob[3] + prob[4];
  }
  if (grade == 2) {
    finalProb = prob[5];
  }
  if (grade == 3) {
    finalProb = prob[6] + prob[7];
  }
  if (grade == 4) {
    finalProb = prob[8] + prob[9];
  }
  if (grade == 5) {
    finalProb = prob[10] + prob[11];
  }
  if (grade == 6) {
    finalProb = prob[12];
  }
  if (grade == 7) {
    finalProb = prob[13];
  }
  if (grade == 8) {
    finalProb = prob[14] + prob[15];
  }
  if (grade == 9) {
    finalProb = prob[16];
  }
  if (grade == 10) {
    finalProb = prob[17];
  }
  if (grade == 11) {
    finalProb = prob[18];
  }
  if (grade == 12) {
    finalProb = prob[19];
  }
  if (grade == 13) {
    finalProb = prob[20];
  }
  return finalProb;
}



function drawClimb(holds) {
  const canvas = document.getElementById("board");
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (boardImage) {
    ctx.drawImage(boardImage, 0, 0, canvas.width, canvas.height);
  }

  for (let i = 0; i < holds.length; i++) {
    const centerX = holds[i][0] * (canvas.width / 144);
    const centerY = canvas.height - holds[i][1] * (canvas.height / 156);
    let x = "";
    if (holds[i][2] == "foot") {
      x = "yellow";
    }
    if (holds[i][2] == "start") {
      x = "lime";
    }
    if (holds[i][2] == "finish") {
      x = "magenta";
    }
    if (holds[i][2] == "middle") {
      x = "cyan";
    }
    ctx.beginPath();
    ctx.arc(centerX, centerY, 12, 0, 2 * Math.PI);
    ctx.strokeStyle = x;
    ctx.lineWidth = 3;
    ctx.stroke();
  }

}

init();
