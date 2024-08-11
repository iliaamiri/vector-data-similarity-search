const { HfInference } = require("@huggingface/inference");
const { ChromaClient } = require("chromadb");
const jobPostings = require("./jobPostings.js");

const collectionName = "job_collection";

const hf = new HfInference("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");
const chroma = new ChromaClient();

main().catch(e => console.error('An error occurred:', e))

async function main() {
    try {
        const collection = await chroma.getOrCreateCollection({
            name: collectionName,
        });

        const uniqueIds = new Set();
        jobPostings.forEach((job, index) => {
            while (uniqueIds.has(job.jobId.toString())) {
                job.jobId = `${job.jobId}_${index}`;
            }
            uniqueIds.add(job.jobId.toString());
        });

        const jobTexts = jobPostings.map(
            (job) =>
                `${job.jobTitle}. ${job.jobDescription}. ${job.jobType}. ${job.location}`,
        );
        const embeddingsData = await generateEmbeddings(jobTexts);

        await collection.add({
            ids: jobPostings.map((job) => job.jobId.toString()),
            documents: jobTexts,
            embeddings: embeddingsData,
        });

        const query = "Creative Studio";

        const filterCriteria = await extractFilterCriteria(query);

        const initialResults = await performSimilaritySearch(
            collection,
            query,
            filterCriteria,
        );

        initialResults.slice(0, 3).forEach((item, index) => {
            console.log(
                `Top ${index + 1} Recommended Job Title ==>, ${item.jobTitle}`,
            );
        });
    } catch (e) {
        console.error("Error:", e);
    }
}

async function performSimilaritySearch(collection, queryTerm, filterCriteria) {
    try {
        const queryEmbedding = await generateEmbeddings([queryTerm]);
        console.log(filterCriteria);
        const results = await collection.query({
            collection: collectionName,
            queryEmbeddings: queryEmbedding,
            n: 3,
        });
        if (!results || results.length === 0) {
            console.log(`No jobs found similar to "${queryTerm}"`);
            return [];
        }
        const topJobs = results.ids[0]
            .map((id, index) => {
                const job = jobPostings.find((item) => item.jobId.toString() === id);
                return {
                    id,
                    score: results.distances[0][index],
                    jobId: job.jobId,
                    jobTitle: job.jobTitle,
                    jobType: job.jobType,
                    jobDescription: job.jobDescription,
                    company: job.company,
                };
            })
            .filter(Boolean);
        return topJobs.sort((a, b) => a.score - b.score);
    } catch (error) {
        console.error("Error during similarity search:", error);
        return [];
    }
}

async function extractFilterCriteria(query) {
    try {
        const criteria = { location: null, jobTitle: null, company: null, jobType: null };
        const labels = ["location", "job title", "company", "job type"];

        const words = query.split(" ");
        for (const word of words) {
            const result = await classifyText(word, labels);
            console.log('result', result);
            const highestScoreLabel = result.labels[0];
            const score = result.scores[0];

            if (score > 0.5) {
                switch (highestScoreLabel) {
                    case "location":
                        criteria.location = word;
                        break;
                    case "job title":
                        criteria.jobTitle = word;
                        break;
                    case "company":
                        criteria.company = word;
                        break;
                    case "job type":
                        criteria.jobType = word;
                        break;
                    default:
                        break;
                }
            }
        }

        return criteria;
    } catch (e) {
        console.error('Error when extracting filter criteria: ', e);
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
        console.error('Error when generating embeddings: ', e);
        throw e;
    }
}

async function classifyText(text, labels) {
    try {
        return hf.request({
            model: "facebook/bart-large-mnli",
            inputs: text,
            parameters: { candidate_labels: labels },
        });
    } catch (e) {
        console.error('Error when classifying text: ', e);
        throw e;
    }
}
