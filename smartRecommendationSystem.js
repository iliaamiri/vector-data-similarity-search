const fs = require("fs");
const pdf = require("pdf-parse");
const { HfInference } = require("@huggingface/inference");
const readline = require("readline");
const { ChromaClient } = require("chromadb");
const jobPostings = require("./jobPostings.js");

const collectionName = "job_collection";

const hf = new HfInference("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");
const chroma = new ChromaClient();

main().catch(e => console.error("An error occurred:", e));

async function main() {
  try {
    await storeEmbeddingsInChromaDB(jobPostings);

    const filePath = await promptUserInput(
        "Enter the path to the resume PDF: ",
    );
    const text = await extractTextFromPDF(filePath);

    const resumeEmbedding = await generateEmbeddings(text);

    const collection = await chroma.getCollection({ name: collectionName });
    const results = await collection.query({
      queryEmbeddings: [resumeEmbedding],
      n: 5, // Get top 5 similar items
    });

    if (results.ids.length <= 0 || results.ids[0].length <= 0) {
      throw new Error("No similar jobs found.");
    }

    results.ids[0].slice(0, 5).forEach((id, index) => {
      const recommendedItem = jobPostings[parseInt(id)];
      console.log(
          `Top ${index + 1} Recommended Item ==> ${recommendedItem.jobTitle}`,
      );
    });
  } catch (e) {
    console.error(e);
    throw e;
  }
}

async function storeEmbeddingsInChromaDB(foodItems) {
  const metadatas = foodItems.map(() => ({})); // Empty metadata objects

  const ids = foodItems.map((_, index) => index.toString());
  const jobTexts = jobPostings.map((job) => job.jobTitle.toLowerCase());
  const embeddingsData = await generateEmbeddings(jobTexts);

  try {
    const collection = await chroma.getOrCreateCollection({
      name: collectionName,
    });

    await collection.add({
      ids: ids,
      documents: jobTexts,
      embeddings: embeddingsData,
      metadatas: metadatas,
    });
    console.log("Stored embeddings in Chroma DB.");
  } catch (e) {
    console.error("Error storing embeddings in Chroma DB:", e);
    throw e;
  }
}

async function promptUserInput(query) {
  try {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    return new Promise((resolve) =>
        rl.question(query, (answer) => {
          rl.close();
          resolve(answer);
        }),
    );
  } catch (e) {
    console.error('Error prompting user input: ', e)
    throw e;
  }
}

async function generateEmbeddings(texts) {
  try {
    return await hf.featureExtraction({
      model: "sentence-transformers/all-MiniLM-L6-v2",
      inputs: texts,
    });
  } catch (e) {
    console.error('Error generating embeddings: ', e);
    throw e;
  }
}

async function extractTextFromPDF(filePath) {
  try {
    const dataBuffer = fs.readFileSync(filePath);
    const data = await pdf(dataBuffer);
    return data.text.replace(/\n/g, " ").replace(/ +/g, " ");
  } catch (e) {
    console.error("Error extracting text from PDF:", e);
    throw e;
  }
}
