export default async function handler(req, res) {
  const backendBase = "http://13.61.8.80:8000";

  const pathParts = Array.isArray(req.query.path)
    ? req.query.path
    : [req.query.path].filter(Boolean);

  const apiPath = `/${pathParts.join("/")}`;

  const queryIndex = req.url.indexOf("?");
  const queryString =
    queryIndex >= 0
      ? req.url.slice(queryIndex)
      : "";

  const targetUrl =
    `${backendBase}${apiPath}${queryString}`;

  try {
    const headers = {
      Accept:
        req.headers.accept ||
        "application/json",
    };

    if (req.headers["content-type"]) {
      headers["Content-Type"] =
        req.headers["content-type"];
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
          : JSON.stringify(req.body ?? {});
    }

    const response =
      await fetch(
        targetUrl,
        options,
      );

    const contentType =
      response.headers.get(
        "content-type",
      ) || "";

    res.status(response.status);

    if (contentType) {
      res.setHeader(
        "Content-Type",
        contentType,
      );
    }

    const body =
      await response.arrayBuffer();

    res.send(
      Buffer.from(body),
    );
  } catch (error) {
    console.error(
      "Backend proxy error:",
      error,
    );

    res.status(502).json({
      detail:
        "Unable to reach flood-world-model API.",
    });
  }
}
