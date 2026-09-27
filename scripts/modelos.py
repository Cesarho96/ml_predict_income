"""Muestra el registry en la terminal: versiones, aliases, origen y veredicto.

    python tasks.py modelos

La UI de MLflow tiene dos modos de registry (el viejo, por "Stage", y el nuevo, por
aliases y tags) y según la versión muestra uno u otro. Esta vista no depende de eso:
lee directo del servidor lo único que importa para decidir una promoción.
"""

from __future__ import annotations

from mlflow import MlflowClient

NOMBRES = ("ingreso_ocupados", "ingreso_no_ocupados")


def main() -> None:
    client = MlflowClient()
    for nombre in NOMBRES:
        modelo = client.get_registered_model(nombre)
        por_version: dict[str, list[str]] = {}
        for alias, version in modelo.aliases.items():
            por_version.setdefault(str(version), []).append("@" + alias)

        print(f"\n{nombre}")
        print(f"  {'ver':>4}  {'aliases':<24} {'origen':<8} {'git':<14} "
              f"{'MdAPE test':>10}  veredicto")
        versiones = client.search_model_versions(f"name = '{nombre}'")
        for mv in sorted(versiones, key=lambda v: int(v.version), reverse=True):
            run = client.get_run(mv.run_id)
            mdape = run.data.metrics.get("test_MdAPE")
            print(f"  {'v' + str(mv.version):>4}  "
                  f"{' '.join(por_version.get(str(mv.version), [])) or '—':<24} "
                  f"{run.data.tags.get('origen', 'manual'):<8} "
                  f"{run.data.tags.get('git_sha', '?')[:14]:<14} "
                  f"{(f'{mdape:.2f}%' if mdape is not None else '—'):>10}  "
                  f"{mv.tags.get('veredicto', '—')}")


if __name__ == "__main__":
    main()
