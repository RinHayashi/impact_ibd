import pandas as pd
from importlib.resources import files, as_file

_ATLAS_API = (
    "visualize_amg_atlas",
    "compute_amg_atlas_coordinates",
    "add_spherical_coordinates",
    "plot_spherical_amg_atlas",
    "AMGAtlasResult",
)


# data loading
def _load_parquet(filename: str) -> pd.DataFrame:
    """Load a bundled Parquet file into a pandas DataFrame."""
    # Locate the resource file in the data subpackage.
    resource_traversable = files("impact_ibd.data").joinpath(filename)
    # Safely convert the Traversable object to a local filesystem path.
    with as_file(resource_traversable) as filepath:
        return pd.read_parquet(filepath)

def load_votu_profile() -> pd.DataFrame:
    """Load and return the bundled vOTU annotation DataFrame.

    Set the first column as the index after loading. Rows are vOTU
    representative sequence names (for example,
    ``PRJEB1220__ERR209533__k89_17697||full``), and columns contain annotations.
    """
    return _load_parquet("vOTU_profile.parquet")

def load_amg_profile() -> pd.DataFrame:
    """Load and return the bundled AMG annotation DataFrame.

    Set the first column as the index after loading. Rows are AMG ORF names
    (for example, ``PRJEB1220__ERR209533__k89_17697__full-cat_2_25``), and
    columns contain annotations.
    """
    return _load_parquet("AMG_profile.parquet")

def load_host_profile() -> pd.DataFrame:
    """Load and return the bundled sample metadata DataFrame.

    Set the first column as the index after loading. Rows are sample names,
    and columns contain metadata.
    """
    return _load_parquet("host_profile.parquet")

def load_prok_profile() -> pd.DataFrame:
    """Load and return the bundled prokaryote annotation DataFrame.

    Set the first column as the index after loading. Rows are taxids (for
    example, ``729``), and columns contain annotations.
    """
    return _load_parquet("Prok_profile.parquet")

def load_amg_abd() -> pd.DataFrame:
    """Load and return the bundled AMG abundance matrix.

    Set the first column as the index after loading. Rows are AMG functions
    (for example, ``K00012``), and columns are sample names.
    """
    return _load_parquet("AMG_abundance.parquet")

def __getattr__(name: str):
    if name in _ATLAS_API:
        from . import atlas as _atlas
        return getattr(_atlas, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_ATLAS_API))


__all__ = [
    "load_votu_profile",
    "load_amg_profile",
    "load_host_profile",
    "load_prok_profile",
    "load_amg_abd",
    *_ATLAS_API,
]
