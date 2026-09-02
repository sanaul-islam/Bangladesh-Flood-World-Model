export default async function handler(req, res) {
  try {
    const response =
      await fetch(
        "http://13.61.8.80:8000/health",
        {
          method: "GET",
          headers: {
            Accept:
              "application/json",
          },
        },
      );

    const contentType =
      response.headers.get(
        "content-type",
      );

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
      "Health proxy error:",
      error,
    );

    res.status(502).json({
      detail:
        "Flood World Model API is unreachable.",
    });
  }
}
