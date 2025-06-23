{
  rustPlatform,
  cargo,
  rustc,
  ...
}:
rustPlatform.buildRustPackage rec {
  name = "mdq";
  version = "0.1.0";
  nativeBuildInputs = [
    cargo
    rustc
  ];
  src = ../../.;
  cargoLock.lockFile = "${src}/Cargo.lock";
}
