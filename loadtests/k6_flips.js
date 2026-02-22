import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "30s", target: 25 },
    { duration: "60s", target: 100 },
    { duration: "30s", target: 0 },
  ],
  thresholds: {
    http_req_duration: ["p(95)<250"],
    checks: ["rate>0.99"],
  },
};

const BASE_URL = __ENV.BASE_URL || "http://localhost:8001";

export default function () {
  const top5 = http.get(`${BASE_URL}/flips/top5?profile=balanced`);
  check(top5, {
    "top5 status 200": (r) => r.status === 200,
    "top5 body has flips": (r) => r.body && r.body.includes("flips"),
  });

  const top = http.get(`${BASE_URL}/flips/top?limit=25&profile=balanced&min_confidence=30`);
  check(top, {
    "top status 200": (r) => r.status === 200,
    "top body has flips": (r) => r.body && r.body.includes("flips"),
  });

  sleep(1);
}

