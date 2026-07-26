"""Static venue table: where a game is played, and whether weather can reach the field.

``collect.weather`` needs two things nflverse's schedule does not reliably give it — a **latitude
and longitude** to ask open-meteo about, and a trustworthy **indoor/outdoor** classification. This
module is that table, and nothing else: no I/O, no network, no clock. It covers every venue that
appears in the 2016-2026 schedules (verified by ``tests/test_collect_weather.py``), which is the
backfill span this phase collects.

Resolution order is **``stadium`` name first, ``stadium_id`` second, home team last**, and that
order is the whole reason this module exists rather than a one-line ``stadium_id -> (lat, lon)``
dict::

    2026_07_PIT_NO   stadium_id='NOR00'  stadium='Stade de France'  roof='dome'
    2026_10_NE_DET   stadium_id='DET00'  stadium='FC Bayern Munich Stadium'  roof='dome'
    2026_01_SF_LA    stadium_id='LAX01'  stadium='Melbourne Cricket Ground'  roof='dome'

For the 2026 international games nflverse carries the **home team's** ``stadium_id`` *and* the home
team's ``roof``, while only ``stadium`` names the real venue. Keying on ``stadium_id`` would place
that Paris game inside the Superdome; trusting ``roof`` would flag eight open-air games as indoor and
discard their weather entirely. (2016-2025 is not affected — those seasons give international games
their own ids, ``LON00``/``LON02``/``MEX00``/``GER00``/``FRA00``/``SAO00``. It is the forward
schedule, i.e. exactly what the pre-lock cron reads, that is provisional.) So the international
venues below are registered **by name only**: giving them a ``stadium_id`` would hijack the real
stadium that owns it.

The team fallback (:data:`HOME_VENUE`) is deliberately restricted to ``location == "Home"`` rows by
:func:`venue_for_game`. On a neutral-site game the home team's stadium is precisely the wrong answer,
and a wrong coordinate is worse than a missing one: a missing one is a null the assembler can see,
while Munich weather reported from Detroit is a plausible-looking lie.

``roof_type`` is a property of the **venue**, where nflverse's ``roof`` is a property of the **game**
(a retractable roof is genuinely open or closed on the day). Both are needed and neither replaces the
other — see :func:`collect.weather.resolve_indoor` for how they combine.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

#: Venue roof classifications.
#:
#: ``outdoor``     open air; weather reaches the field.
#: ``dome``        permanently enclosed; a game here never sees weather.
#: ``retractable`` has a roof that opens, so only the game's own ``roof`` value settles it.
ROOF_TYPES: tuple[str, ...] = ("outdoor", "dome", "retractable")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class Venue:
    """One playing venue. Coordinates are the playing surface, to ~4 decimal places (~10 m)."""

    venue_id: str
    name: str
    roof_type: Literal["outdoor", "dome", "retractable"]
    latitude: float
    longitude: float
    #: PFR stadium codes nflverse files this venue under. Empty for venues that only ever appear
    #: under *another* stadium's id (the 2026 international games — see the module docstring).
    stadium_ids: tuple[str, ...] = ()
    #: Other ``stadium`` spellings nflverse has used (sponsor renames, translations).
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.roof_type not in ROOF_TYPES:
            raise ValueError(f"{self.venue_id}: roof_type {self.roof_type!r} not in {ROOF_TYPES}")
        if not -90 <= self.latitude <= 90 or not -180 <= self.longitude <= 180:
            raise ValueError(f"{self.venue_id}: ({self.latitude}, {self.longitude}) is not on Earth")

    @property
    def dome(self) -> bool:
        """Permanently enclosed — the ``dome: bool`` the ticket asks for.

        A **retractable** venue is not a dome by this flag even when its roof happens to be shut for
        a given game; that is a per-game fact and lives on the schedule's ``roof``, not here.
        """
        return self.roof_type == "dome"


def _v(
    venue_id: str,
    name: str,
    roof_type: str,
    latitude: float,
    longitude: float,
    *,
    stadium_ids: tuple[str, ...] = (),
    aliases: tuple[str, ...] = (),
) -> Venue:
    return Venue(
        venue_id=venue_id,
        name=name,
        roof_type=roof_type,  # type: ignore[arg-type]
        latitude=latitude,
        longitude=longitude,
        stadium_ids=stadium_ids,
        aliases=aliases,
    )


_VENUES: tuple[Venue, ...] = (
    # --- current NFL venues --------------------------------------------------------------------
    _v("ari_state_farm", "State Farm Stadium", "retractable", 33.5276, -112.2626,
       stadium_ids=("PHO00",), aliases=("University of Phoenix Stadium",)),
    _v("atl_mercedes_benz", "Mercedes-Benz Stadium", "retractable", 33.7554, -84.4008,
       stadium_ids=("ATL97",)),
    _v("bal_mt_bank", "M&T Bank Stadium", "outdoor", 39.2780, -76.6227, stadium_ids=("BAL00",)),
    _v("buf_highmark", "Highmark Stadium", "outdoor", 42.7738, -78.7870,
       stadium_ids=("BUF00",), aliases=("New Era Field",)),
    _v("car_bank_of_america", "Bank of America Stadium", "outdoor", 35.2258, -80.8528,
       stadium_ids=("CAR00",)),
    _v("chi_soldier_field", "Soldier Field", "outdoor", 41.8623, -87.6167,
       stadium_ids=("CHI98",)),
    _v("cin_paycor", "Paycor Stadium", "outdoor", 39.0955, -84.5161,
       stadium_ids=("CIN00",), aliases=("Paul Brown Stadium",)),
    _v("cle_huntington_bank", "Huntington Bank Field", "outdoor", 41.5061, -81.6995,
       stadium_ids=("CLE00",), aliases=("FirstEnergy Stadium",)),
    _v("dal_att", "AT&T Stadium", "retractable", 32.7473, -97.0945, stadium_ids=("DAL00",)),
    _v("den_empower_field", "Empower Field at Mile High", "outdoor", 39.7439, -105.0201,
       stadium_ids=("DEN00",), aliases=("Sports Authority Field at Mile High",)),
    _v("det_ford_field", "Ford Field", "dome", 42.3400, -83.0456, stadium_ids=("DET00",)),
    _v("gb_lambeau", "Lambeau Field", "outdoor", 44.5013, -88.0622, stadium_ids=("GNB00",)),
    _v("hou_nrg", "NRG Stadium", "retractable", 29.6847, -95.4107,
       stadium_ids=("HOU00",), aliases=("Reliant Stadium",)),
    _v("ind_lucas_oil", "Lucas Oil Stadium", "retractable", 39.7601, -86.1639,
       stadium_ids=("IND00",)),
    _v("jax_everbank", "EverBank Stadium", "outdoor", 30.3239, -81.6373,
       stadium_ids=("JAX00",), aliases=("EverBank Field", "TIAA Bank Stadium")),
    _v("kc_arrowhead", "GEHA Field at Arrowhead Stadium", "outdoor", 39.0489, -94.4839,
       stadium_ids=("KAN00",), aliases=("Arrowhead Stadium",)),
    # SoFi's canopy is fixed but its sides are open. nflverse calls it a dome and this table agrees:
    # the field is sheltered from rain and the recorded temp/wind are not the outdoor ones.
    _v("la_sofi", "SoFi Stadium", "dome", 33.9535, -118.3392, stadium_ids=("LAX01",)),
    _v("lv_allegiant", "Allegiant Stadium", "dome", 36.0909, -115.1833, stadium_ids=("VEG00",)),
    # A canopy over the seating bowl only; the field itself is open to the sky.
    _v("mia_hard_rock", "Hard Rock Stadium", "outdoor", 25.9580, -80.2389, stadium_ids=("MIA00",)),
    _v("min_us_bank", "U.S. Bank Stadium", "dome", 44.9738, -93.2578, stadium_ids=("MIN01",)),
    _v("ne_gillette", "Gillette Stadium", "outdoor", 42.0909, -71.2643, stadium_ids=("BOS00",)),
    _v("no_superdome", "Caesars Superdome", "dome", 29.9511, -90.0812,
       stadium_ids=("NOR00",), aliases=("Mercedes-Benz Superdome",)),
    _v("nyc_metlife", "MetLife Stadium", "outdoor", 40.8135, -74.0745, stadium_ids=("NYC01",)),
    _v("phi_lincoln_financial", "Lincoln Financial Field", "outdoor", 39.9008, -75.1675,
       stadium_ids=("PHI00",)),
    _v("pit_acrisure", "Acrisure Stadium", "outdoor", 40.4468, -80.0158,
       stadium_ids=("PIT00",), aliases=("Heinz Field",)),
    _v("sea_lumen", "Lumen Field", "outdoor", 47.5952, -122.3316,
       stadium_ids=("SEA00",), aliases=("CenturyLink Field",)),
    _v("sf_levis", "Levi's Stadium", "outdoor", 37.4030, -121.9698, stadium_ids=("SFO01",)),
    _v("tb_raymond_james", "Raymond James Stadium", "outdoor", 27.9759, -82.5033,
       stadium_ids=("TAM00",)),
    _v("ten_nissan", "Nissan Stadium", "outdoor", 36.1665, -86.7713, stadium_ids=("NAS00",)),
    _v("was_northwest", "Northwest Stadium", "outdoor", 38.9076, -76.8645,
       stadium_ids=("WAS00",), aliases=("FedExField",)),
    # --- retired / relocated venues still in the 2016-2025 backfill span ------------------------
    _v("atl_georgia_dome", "Georgia Dome", "dome", 33.7576, -84.4008, stadium_ids=("ATL00",)),
    _v("la_coliseum", "Los Angeles Memorial Coliseum", "outdoor", 34.0141, -118.2879,
       stadium_ids=("LAX99",)),
    _v("lac_dignity_health", "StubHub Center", "outdoor", 33.8644, -118.2611,
       stadium_ids=("LAX97",)),
    _v("oak_coliseum", "Oakland-Alameda County Coliseum", "outdoor", 37.7516, -122.2005,
       stadium_ids=("OAK00",), aliases=("Ring Central Coliseum",)),
    _v("sd_qualcomm", "Qualcomm Stadium", "outdoor", 32.7831, -117.1196, stadium_ids=("SDG00",)),
    # --- international venues ------------------------------------------------------------------
    # Those with their own PFR code (how 2016-2025 files them)...
    _v("lon_wembley", "Wembley Stadium", "outdoor", 51.5560, -0.2795, stadium_ids=("LON00",)),
    _v("lon_twickenham", "Twickenham Stadium", "outdoor", 51.4560, -0.3417,
       stadium_ids=("LON01",)),
    _v("lon_tottenham", "Tottenham Hotspur Stadium", "outdoor", 51.6043, -0.0665,
       stadium_ids=("LON02",), aliases=("Tottenham Stadium",)),
    _v("mex_azteca", "Estadio Azteca", "outdoor", 19.3029, -99.1505,
       stadium_ids=("MEX00",), aliases=("Azteca Stadium", "Estadio Banorte")),
    _v("mun_allianz", "Allianz Arena", "outdoor", 48.2188, 11.6247,
       stadium_ids=("GER00",), aliases=("FC Bayern Munich Stadium",)),
    _v("fra_deutsche_bank_park", "Deutsche Bank Park", "outdoor", 50.0685, 8.6455,
       stadium_ids=("FRA00",)),
    _v("sao_corinthians", "Arena Corinthians", "outdoor", -23.5453, -46.4742,
       stadium_ids=("SAO00",), aliases=("Neo Quimica Arena",)),
    # ...and those that appear ONLY under a home team's id (2026). No stadium_ids on purpose: one
    # here would steal the code from the real stadium that owns it.
    _v("mel_mcg", "Melbourne Cricket Ground", "outdoor", -37.8200, 144.9834),
    _v("rio_maracana", "Maracana Stadium", "outdoor", -22.9121, -43.2302,
       aliases=("Estadio Maracana",)),
    _v("par_stade_de_france", "Stade de France", "outdoor", 48.9245, 2.3601),
    _v("mad_bernabeu", "Santiago Bernabeu", "retractable", 40.4531, -3.6883,
       aliases=("Bernabeu", "Estadio Santiago Bernabeu")),
    _v("dub_croke_park", "Croke Park", "outdoor", 53.3607, -6.2512),
)

#: Every known venue, keyed by ``venue_id``.
VENUES: dict[str, Venue] = {v.venue_id: v for v in _VENUES}

#: Where each franchise plays *today*. Only ever consulted for a ``location == "Home"`` row whose
#: stadium name and id are both unrecognised — see :func:`venue_for_game`.
HOME_VENUE: dict[str, str] = {
    "ARI": "ari_state_farm", "ATL": "atl_mercedes_benz", "BAL": "bal_mt_bank",
    "BUF": "buf_highmark", "CAR": "car_bank_of_america", "CHI": "chi_soldier_field",
    "CIN": "cin_paycor", "CLE": "cle_huntington_bank", "DAL": "dal_att",
    "DEN": "den_empower_field", "DET": "det_ford_field", "GB": "gb_lambeau", "HOU": "hou_nrg",
    "IND": "ind_lucas_oil", "JAX": "jax_everbank", "KC": "kc_arrowhead", "LA": "la_sofi",
    "LAC": "la_sofi", "LV": "lv_allegiant", "MIA": "mia_hard_rock", "MIN": "min_us_bank",
    "NE": "ne_gillette", "NO": "no_superdome", "NYG": "nyc_metlife", "NYJ": "nyc_metlife",
    "OAK": "oak_coliseum", "PHI": "phi_lincoln_financial", "PIT": "pit_acrisure",
    "SD": "sd_qualcomm", "SEA": "sea_lumen", "SF": "sf_levis", "TB": "tb_raymond_james",
    "TEN": "ten_nissan", "WAS": "was_northwest",
}


def normalize_name(value: str) -> str:
    """A stadium name reduced to comparable form: accents stripped, non-alphanumerics dropped.

    ``"Maracanã Stadium"``, ``"Maracana Stadium"`` and ``"maracana  stadium"`` all collapse to
    ``"maracanastadium"``, which is what lets the aliases above stay short.
    """
    folded = unicodedata.normalize("NFKD", value or "")
    ascii_only = folded.encode("ascii", "ignore").decode("ascii")
    return _NON_ALNUM.sub("", ascii_only.lower())


def _index_names() -> dict[str, Venue]:
    index: dict[str, Venue] = {}
    for venue in _VENUES:
        for spelling in (venue.name, *venue.aliases):
            key = normalize_name(spelling)
            if key in index and index[key] is not venue:  # pragma: no cover - guards a bad edit
                raise ValueError(f"stadium name {spelling!r} maps to two venues")
            index[key] = venue
    return index


def _index_ids() -> dict[str, Venue]:
    index: dict[str, Venue] = {}
    for venue in _VENUES:
        for code in venue.stadium_ids:
            if code in index:  # pragma: no cover - guards a bad edit
                raise ValueError(f"stadium_id {code!r} maps to two venues")
            index[code] = venue
    return index


_BY_NAME: dict[str, Venue] = _index_names()
_BY_STADIUM_ID: dict[str, Venue] = _index_ids()


def venue_by_name(stadium: str | None) -> Venue | None:
    """The venue nflverse's ``stadium`` string names, or ``None`` if it is unknown."""
    if not stadium:
        return None
    return _BY_NAME.get(normalize_name(stadium))


def venue_by_stadium_id(stadium_id: str | None) -> Venue | None:
    """The venue owning a PFR ``stadium_id``, or ``None``."""
    if not stadium_id:
        return None
    return _BY_STADIUM_ID.get(stadium_id.strip())


def venue_for_team(team: str | None) -> Venue | None:
    """A franchise's current home venue, or ``None`` for an unknown abbreviation."""
    if not team:
        return None
    venue_id = HOME_VENUE.get(team.strip().upper())
    return VENUES.get(venue_id) if venue_id else None


def venue_for_game(
    *,
    stadium: str | None = None,
    stadium_id: str | None = None,
    home_team: str | None = None,
    location: str | None = None,
) -> Venue | None:
    """Where a scheduled game is played, or ``None`` when the venue cannot be identified.

    Name first, then id, then — **only for a game nflverse marks ``location == "Home"``** — the home
    team's current stadium. Returning ``None`` is a real answer: the caller records a null
    coordinate and skips the forecast, which is strictly better than fetching the weather at a
    stadium the game is not being played in.
    """
    return (
        venue_by_name(stadium)
        or venue_by_stadium_id(stadium_id)
        or (venue_for_team(home_team) if (location or "").strip().lower() == "home" else None)
    )
