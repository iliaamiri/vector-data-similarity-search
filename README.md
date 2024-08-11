# Summary
This is an experimentation with vector data and similarity searching (Cosine similarity) 
using open-source models and ChromaDB.

# Step-by-step setup guide
Follow each step in order to setup and run.

### Clone
```bash
git clone https://github.com/iliaamiri/vector-data-similarity-search.git 
```

### Install packages
```bash
npm i
```

### Run ChromaDB docker container
```bash
docker run --name chroma-container-name -d -p 8000:8000 chromadb/chroma
```

### Grab an API key
Sign up on [Huggingface](https://huggingface.co/) and generate a temporary token.
Then use the token at the beginning of each js files.

### Run 
Food recommendation example:
```bash
node foodRecommendationSystem.js
```

Job recommendation system example:
```bash
node jobRecommendationSystem.js
node smartRecommendationSystem.js
```

