export interface TrustedPlace {
  id: string;
  name: string;
  address: string;
  latitude: number;
  longitude: number;
  aliases: string[];
  source?: string;
}

/*
 * Curated locations where a current/authoritative
 * location should take priority over generic geocoding.
 */
export const TRUSTED_PLACES: TrustedPlace[] = [
  {
    id: "bracu-main-campus",

    name: "BRAC University",

    address:
      "Kha 224 Pragati Sarani, Merul Badda, Dhaka 1212, Bangladesh",

    /*
     * Approximate current Merul Badda campus coordinates.
     * Keep configurable so they can be updated when a
     * more authoritative coordinate is available.
     */
    latitude: 23.7725,
    longitude: 90.4254,

    aliases: [
      "brac university",
      "bracu",
      "brac u",
      "brac university campus",
      "bracu campus",
      "brac university merul badda",
      "bracu merul badda",
    ],

    source:
      "BRAC University official campus address",
  },
];

function normalizeQuery(
  value: string,
): string {
  return value
    .toLowerCase()
    .normalize("NFKD")
    .replace(
      /[\u0300-\u036f]/g,
      "",
    )
    .replace(
      /[^a-z0-9\u0980-\u09ff\s]/g,
      " ",
    )
    .replace(
      /\s+/g,
      " ",
    )
    .trim();
}

export function findTrustedPlaces(
  query: string,
): TrustedPlace[] {
  const normalized =
    normalizeQuery(query);

  if (!normalized) {
    return [];
  }

  return TRUSTED_PLACES.filter(
    (place) => {
      const normalizedName =
        normalizeQuery(
          place.name,
        );

      const normalizedAliases =
        place.aliases.map(
          normalizeQuery,
        );

      /*
       * Exact name match.
       */
      if (
        normalized ===
        normalizedName
      ) {
        return true;
      }

      /*
       * Exact alias match.
       */
      if (
        normalizedAliases.includes(
          normalized,
        )
      ) {
        return true;
      }

      /*
       * Handle queries such as:
       *
       * "BRAC University Dhaka"
       * "BRAC University Merul Badda"
       */
      if (
        normalized.includes(
          normalizedName,
        )
      ) {
        return true;
      }

      return normalizedAliases.some(
        (alias) =>
          alias.length >= 4 &&
          normalized.includes(alias),
      );
    },
  );
}
