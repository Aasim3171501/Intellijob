/**
 * Curated learning-resource library keyed by skill name.
 *
 * Every entry carries a small set of *key sub-topics* and three verified
 * resource links: official documentation, a tutorial/guide, and an
 * interactive learning path. Skills not in the library fall back to safe
 * search links so a rare milestone always yields something useful.
 */

export type ResourceKind = 'documentation' | 'tutorial' | 'interactive';

export interface LearningResource {
  title: string;
  url: string;
  domain: string;
  kind: ResourceKind;
}

interface SkillLearning {
  topics: string[];
  resources: LearningResource[];
}

const LIBRARY: Record<string, SkillLearning> = {
  'spring framework': {
    topics: ['Spring Core & IoC', 'Spring Boot', 'Spring Data JPA', 'REST Controllers'],
    resources: [
      { title: 'Spring Framework Documentation', url: 'https://spring.io/projects/spring-framework', domain: 'spring.io', kind: 'documentation' },
      { title: 'Baeldung Spring Tutorials', url: 'https://www.baeldung.com/spring-tutorial', domain: 'baeldung.com', kind: 'tutorial' },
      { title: 'Java Spring Boot Roadmap', url: 'https://roadmap.sh/java-spring-boot', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'python': {
    topics: ['Syntax & Data Types', 'Functions & Classes', 'Standard Library', 'Packaging'],
    resources: [
      { title: 'Python Official Documentation', url: 'https://docs.python.org/3/', domain: 'docs.python.org', kind: 'documentation' },
      { title: 'W3Schools Python Tutorial', url: 'https://www.w3schools.com/python/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: 'Python Developer Roadmap', url: 'https://roadmap.sh/python', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'java': {
    topics: ['Core Syntax & OOP', 'Collections Framework', 'Streams & Lambdas', 'Maven & Build'],
    resources: [
      { title: 'Java Documentation (Oracle)', url: 'https://docs.oracle.com/en/java/', domain: 'docs.oracle.com', kind: 'documentation' },
      { title: 'W3Schools Java Tutorial', url: 'https://www.w3schools.com/java/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: 'Java Developer Roadmap', url: 'https://roadmap.sh/java', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'javascript': {
    topics: ['Variables & Scope', 'Functions & Closures', 'DOM Manipulation', 'ES Modules'],
    resources: [
      { title: 'MDN JavaScript Reference', url: 'https://developer.mozilla.org/en-US/docs/Web/JavaScript', domain: 'developer.mozilla.org', kind: 'documentation' },
      { title: 'W3Schools JavaScript Tutorial', url: 'https://www.w3schools.com/js/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: 'JavaScript Roadmap', url: 'https://roadmap.sh/javascript', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'typescript': {
    topics: ['Type System & Inference', 'Interfaces & Generics', 'TypeScript Config', 'Tooling & ESLint'],
    resources: [
      { title: 'TypeScript Documentation', url: 'https://www.typescriptlang.org/docs/', domain: 'typescriptlang.org', kind: 'documentation' },
      { title: 'W3Schools TypeScript Tutorial', url: 'https://www.w3schools.com/typescript/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: 'TypeScript Roadmap', url: 'https://roadmap.sh/typescript', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'sql': {
    topics: ['SELECT & Filtering', 'Joins & Aggregates', 'Subqueries & CTEs', 'Indexing'],
    resources: [
      { title: 'PostgreSQL SQL Reference', url: 'https://www.postgresql.org/docs/current/sql.html', domain: 'postgresql.org', kind: 'documentation' },
      { title: 'W3Schools SQL Tutorial', url: 'https://www.w3schools.com/sql/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: 'SQL Roadmap', url: 'https://roadmap.sh/sql', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'postgresql': {
    topics: ['Tables & Constraints', 'SQL Queries', 'Transactions & Locks', 'Indexes & Performance'],
    resources: [
      { title: 'PostgreSQL Official Docs', url: 'https://www.postgresql.org/docs/', domain: 'postgresql.org', kind: 'documentation' },
      { title: 'PostgreSQL Tutorial', url: 'https://www.postgresqltutorial.com/', domain: 'postgresqltutorial.com', kind: 'tutorial' },
      { title: 'PostgreSQL DBA Roadmap', url: 'https://roadmap.sh/postgresql-dba', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'fastapi': {
    topics: ['Routing & Path Params', 'Pydantic Models', 'Dependency Injection', 'Async Endpoints'],
    resources: [
      { title: 'FastAPI Documentation', url: 'https://fastapi.tiangolo.com/', domain: 'fastapi.tiangolo.com', kind: 'documentation' },
      { title: 'freeCodeCamp FastAPI Guide', url: 'https://www.freecodecamp.org/news/fastapi-python/', domain: 'freecodecamp.org', kind: 'tutorial' },
      { title: 'Python Developer Roadmap', url: 'https://roadmap.sh/python', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'django': {
    topics: ['Models & Migrations', 'Views & URLs', 'Templates & Forms', 'Authentication'],
    resources: [
      { title: 'Django Documentation', url: 'https://docs.djangoproject.com/', domain: 'docs.djangoproject.com', kind: 'documentation' },
      { title: 'W3Schools Django Tutorial', url: 'https://www.w3schools.com/django/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: 'Python Developer Roadmap', url: 'https://roadmap.sh/python', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'docker': {
    topics: ['Images & Containers', 'Dockerfile & Build', 'Volumes & Networks', 'Compose'],
    resources: [
      { title: 'Docker Documentation', url: 'https://docs.docker.com/', domain: 'docs.docker.com', kind: 'documentation' },
      { title: 'The Docker Handbook', url: 'https://www.freecodecamp.org/news/the-docker-handbook/', domain: 'freecodecamp.org', kind: 'tutorial' },
      { title: 'Docker Roadmap', url: 'https://roadmap.sh/docker', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'kubernetes': {
    topics: ['Pods & Deployments', 'Services & Networking', 'ConfigMaps & Secrets', 'Helm'],
    resources: [
      { title: 'Kubernetes Documentation', url: 'https://kubernetes.io/docs/', domain: 'kubernetes.io', kind: 'documentation' },
      { title: 'freeCodeCamp Kubernetes Guides', url: 'https://www.freecodecamp.org/news/tag/kubernetes/', domain: 'freecodecamp.org', kind: 'tutorial' },
      { title: 'Kubernetes Roadmap', url: 'https://roadmap.sh/kubernetes', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'aws': {
    topics: ['EC2 & VPC', 'S3 Storage', 'IAM & Security', 'Lambda & Serverless'],
    resources: [
      { title: 'AWS Documentation', url: 'https://docs.aws.amazon.com/', domain: 'docs.aws.amazon.com', kind: 'documentation' },
      { title: 'AWS Training & Certification', url: 'https://aws.amazon.com/training/', domain: 'aws.amazon.com', kind: 'tutorial' },
      { title: 'AWS Roadmap', url: 'https://roadmap.sh/aws', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'react': {
    topics: ['Components & Props', 'State & Hooks', 'Effects & Data Fetching', 'Routing'],
    resources: [
      { title: 'React Documentation', url: 'https://react.dev/', domain: 'react.dev', kind: 'documentation' },
      { title: 'W3Schools React Tutorial', url: 'https://www.w3schools.com/react/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: 'React Developer Roadmap', url: 'https://roadmap.sh/react', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'node.js': {
    topics: ['Event Loop & Async', 'Modules & NPM', 'Express & APIs', 'Error Handling'],
    resources: [
      { title: 'Node.js Documentation', url: 'https://nodejs.org/en/docs', domain: 'nodejs.org', kind: 'documentation' },
      { title: 'W3Schools Node.js Tutorial', url: 'https://www.w3schools.com/nodejs/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: 'Node.js Roadmap', url: 'https://roadmap.sh/nodejs', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'git': {
    topics: ['Commits & Branches', 'Merging & Rebasing', 'Remotes & Collaboration', 'GitHub Workflows'],
    resources: [
      { title: 'Git Documentation', url: 'https://git-scm.com/doc', domain: 'git-scm.com', kind: 'documentation' },
      { title: 'W3Schools Git Tutorial', url: 'https://www.w3schools.com/git/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: 'Git & GitHub Roadmap', url: 'https://roadmap.sh/git-and-github', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'machine learning': {
    topics: ['Supervised Learning', 'Model Evaluation', 'Feature Engineering', 'Scikit-learn'],
    resources: [
      { title: 'Scikit-learn Documentation', url: 'https://scikit-learn.org/stable/', domain: 'scikit-learn.org', kind: 'documentation' },
      { title: 'freeCodeCamp ML for Beginners', url: 'https://www.freecodecamp.org/news/machine-learning-for-beginners/', domain: 'freecodecamp.org', kind: 'tutorial' },
      { title: 'AI & Data Scientist Roadmap', url: 'https://roadmap.sh/ai-data-scientist', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'data structures & algorithms': {
    topics: ['Arrays & Linked Lists', 'Trees & Graphs', 'Sorting & Searching', 'Dynamic Programming'],
    resources: [
      { title: 'GeeksforGeeks DSA Guide', url: 'https://www.geeksforgeeks.org/data-structures/', domain: 'geeksforgeeks.org', kind: 'documentation' },
      { title: 'W3Schools DSA Tutorial', url: 'https://www.w3schools.com/dsa/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: 'freeCodeCamp DSA Course', url: 'https://www.freecodecamp.org/news/learn-data-structures-and-algorithms/', domain: 'freecodecamp.org', kind: 'interactive' },
    ],
  },
  'c#': {
    topics: ['Syntax & OOP', 'LINQ', 'Async / Await', '.NET Runtime'],
    resources: [
      { title: 'C# Documentation', url: 'https://learn.microsoft.com/en-us/dotnet/csharp/', domain: 'learn.microsoft.com', kind: 'documentation' },
      { title: 'W3Schools C# Tutorial', url: 'https://www.w3schools.com/cs/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: 'C# Learning Path', url: 'https://learn.microsoft.com/en-us/training/paths/get-started-c-sharp-part-1/', domain: 'learn.microsoft.com', kind: 'interactive' },
    ],
  },
  '.net': {
    topics: ['.NET SDK & CLI', 'ASP.NET Core', 'Dependency Injection', 'EF Core'],
    resources: [
      { title: '.NET Documentation', url: 'https://learn.microsoft.com/en-us/dotnet/', domain: 'learn.microsoft.com', kind: 'documentation' },
      { title: 'W3Schools C# Tutorial', url: 'https://www.w3schools.com/cs/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: '.NET Learning Path', url: 'https://learn.microsoft.com/en-us/training/browse/?products=dotnet', domain: 'learn.microsoft.com', kind: 'interactive' },
    ],
  },
  'mongodb': {
    topics: ['Documents & Collections', 'CRUD Operations', 'Indexes', 'Aggregation Pipeline'],
    resources: [
      { title: 'MongoDB Documentation', url: 'https://www.mongodb.com/docs/', domain: 'mongodb.com', kind: 'documentation' },
      { title: 'W3Schools MongoDB Tutorial', url: 'https://www.w3schools.com/mongodb/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: 'Backend Developer Roadmap', url: 'https://roadmap.sh/backend', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'redis': {
    topics: ['Data Types', 'Caching Patterns', 'Pub/Sub', 'Persistence'],
    resources: [
      { title: 'Redis Documentation', url: 'https://redis.io/docs/', domain: 'redis.io', kind: 'documentation' },
      { title: 'Redis Learn Hub', url: 'https://redis.io/learn/', domain: 'redis.io', kind: 'tutorial' },
      { title: 'Backend Developer Roadmap', url: 'https://roadmap.sh/backend', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'go': {
    topics: ['Syntax & Packages', 'Concurrency & Goroutines', 'Channels', 'Testing'],
    resources: [
      { title: 'Go Documentation', url: 'https://go.dev/doc/', domain: 'go.dev', kind: 'documentation' },
      { title: 'W3Schools Go Tutorial', url: 'https://www.w3schools.com/go/', domain: 'w3schools.com', kind: 'tutorial' },
      { title: 'Go Developer Roadmap', url: 'https://roadmap.sh/golang', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
  'terraform': {
    topics: ['HCL Syntax', 'Providers & Resources', 'State Management', 'Modules'],
    resources: [
      { title: 'Terraform Documentation', url: 'https://developer.hashicorp.com/terraform/docs', domain: 'developer.hashicorp.com', kind: 'documentation' },
      { title: 'Terraform Tutorials', url: 'https://developer.hashicorp.com/terraform/tutorials', domain: 'developer.hashicorp.com', kind: 'tutorial' },
      { title: 'DevOps Roadmap', url: 'https://roadmap.sh/devops', domain: 'roadmap.sh', kind: 'interactive' },
    ],
  },
};

const GENERIC_TOPICS = ['Hands-on Practice', 'Real-World Project', 'Best Practices'];

/** Normalise a skill label for dictionary lookup. */
function normalizeSkill(skill: string): string {
  return skill
    .trim()
    .toLowerCase()
    .replace(/^(learn|master|deepen)\s+/, '')
    .replace(/[:.]+$/, '');
}

/** Build safe search links for skills not in the curated library. */
export function fallbackResources(skill: string): LearningResource[] {
  const query = encodeURIComponent(skill.trim());
  return [
    {
      title: `W3Schools search for ${skill}`,
      url: `https://www.w3schools.com/search/search.asp?query=${query}`,
      domain: 'w3schools.com',
      kind: 'tutorial',
    },
    {
      title: `MDN search for ${skill}`,
      url: `https://developer.mozilla.org/search?q=${query}`,
      domain: 'developer.mozilla.org',
      kind: 'documentation',
    },
    {
      title: `roadmap.sh search for ${skill}`,
      url: `https://roadmap.sh/search?q=${query}`,
      domain: 'roadmap.sh',
      kind: 'interactive',
    },
  ];
}

/** Recommended resources for a skill — curated, else fuzzy, else search links. */
export function getLearningResources(skill: string): LearningResource[] {
  const key = normalizeSkill(skill);
  if (!key) return fallbackResources(skill);

  const exact = LIBRARY[key];
  if (exact && exact.resources.length) return exact.resources;

  // Fuzzy: a library key that contains the skill, or vice versa
  // (e.g. "DSA" -> "data structures & algorithms", "Spring" -> "spring framework").
  const fuzzyKey = Object.keys(LIBRARY).find(
    (k) => key.includes(k) || k.includes(key),
  );
  if (fuzzyKey) return LIBRARY[fuzzyKey].resources;

  return fallbackResources(skill);
}

/** Key sub-topics for a skill, used in the step-detail modal. */
export function getKeyTopics(skill: string): string[] {
  const key = normalizeSkill(skill);
  if (!key) return GENERIC_TOPICS;

  const exact = LIBRARY[key];
  if (exact && exact.topics.length) return exact.topics;

  const fuzzyKey = Object.keys(LIBRARY).find(
    (k) => key.includes(k) || k.includes(key),
  );
  if (fuzzyKey) return LIBRARY[fuzzyKey].topics;

  return GENERIC_TOPICS;
}