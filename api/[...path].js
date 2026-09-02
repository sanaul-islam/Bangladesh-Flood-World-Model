export default async function handler(req, res) {
  const backendBase = "http://13.61.8.80:8000";

  const pathValue = req.query.path;

  const parts = Array.isArray(pathValue)
    ? pathValue
    : pathValue
      ? [pathValue]
      : [];

  const path = `/${parts.join("/")}`;

  const questionMark = req.url.indexOf("?");

  const queryString =
    questionMark >= 0
      ? req.url.slice(questionMark)
      : "";

  const targetUrl =
    `${backendBase}${path}${queryString}`;

  try {
    const headers = {
      Accept:
        req.headers.accept ||
        "application/json",
    };

    const contentType =
      req.headers["content-type"];

    if (typeof contentType === "string") {
      headers["Content-Type"] =
        contentType;
    }

    const options = {
      method: req.method,
      headers,
    };

    if (
      req.method !== "GET" &&
      req.method !== "HEAD"
    ) {
      options.body =
        typeof req.body === "string"
          ? req.body
          : JSON.stringify(
              req.body ?? {},
            );
    }

    const response =
      await fetch(
        targetUrl,
        options,
      );

    const responseType =
      response.headers.get(
        "content-type",
      );

    res.status(response.status);

    if (responseType) {
      res.setHeader(
        "Content-Type",
        responseType,
      );
    }

    const body =
      await response.arrayBuffer();

    res.send(
      Buffer.from(body),
    );
  } catch (error) {
    console.error(
      "Flood API proxy error:",
      error,
    );

    res.status(502).json({
      detail:
        "Flood World Model API is unreachable.",
    });
  }
}
