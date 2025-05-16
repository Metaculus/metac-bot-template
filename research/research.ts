import fs from "fs";
import { format } from "date-fns";

// list files in ../benchmarks/
const files = fs.readdirSync("../benchmarks/");

// filter only json files
const jsonFiles = files.filter((file) => file.endsWith(".json")).sort();

// take latest file (last element of jsonFiles)
const latestFile = jsonFiles[jsonFiles.length - 1];

// read all files
const runs = jsonFiles.map((file) => {
  const data = fs.readFileSync(`../benchmarks/${file}`, "utf8");
  return JSON.parse(data) as Run;
});

// full runs
// filter out runs with < 30 questions
const fullRuns = runs.filter((run) => {
  return run.some((benchmark) => benchmark.num_input_questions >= 30);
});

// calculate average baseline score for each benchmark
const scoreboard = fullRuns
  .flatMap((benchmark) => {
    return benchmark.map((benchmark) => {
      const baseline = averageBaselineScore(benchmark.forecast_reports);
      const community = perfectPredictorScore(benchmark.forecast_reports);
      const score = Math.round(10000 * (baseline / community)) / 100;

      return {
        bot: benchmark.forecast_bot_class_name,
        date: format(new Date(benchmark.timestamp), "MMM do, h:mmaaa"),
        score,
        baseline,
        community,
      };
    });
  })
  .sort((a, b) => b.score - a.score)
  .slice(0, 10);

console.log(scoreboard);

type Run = Benchmark[];

type Benchmark = {
  explicit_name: string | null;
  explicit_description: string | null;
  forecast_bot_class_name: string;
  num_input_questions: number;
  timestamp: string;
  time_taken_in_minutes: number;
  total_cost: number;
  git_commit_hash: string;
  forecast_bot_config: Record<string, any>;
  code: string;
  forecast_reports: ForecastReport[];
};

type ForecastReport = {
  question: {
    community_prediction_at_access_time: number;
    // ...other fields...
  };
  prediction: number;
  // ...other fields...
};

function expectedBaselineScore(report: ForecastReport): number {
  const c = report.question.community_prediction_at_access_time;
  const p = report.prediction;
  if (c == null || p == null || p <= 0 || p >= 1) {
    return NaN;
  }
  return (
    100 * (c * (Math.log2(p) + 1.0) + (1.0 - c) * (Math.log2(1.0 - p) + 1.0))
  );
}

function averageBaselineScore(reports: ForecastReport[]): number {
  return (
    reports.reduce((acc, report) => acc + expectedBaselineScore(report), 0) /
    reports.length
  );
}

function perfectPredictorScore(reports: ForecastReport[]): number {
  function scoreForC(c: number): number {
    // Avoid log(0) and log(1) issues
    if (c == null || c <= 0 || c >= 1) return NaN;
    return 100 * (c * (Math.log2(c) + 1) + (1 - c) * (Math.log2(1 - c) + 1));
  }
  const scores = reports
    .map((r) => scoreForC(r.question.community_prediction_at_access_time))
    .filter((s) => !isNaN(s));
  return scores.reduce((a, b) => a + b, 0) / scores.length;
}
