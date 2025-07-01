{
  self,
  pkgs,
  lib,
  inputs,
  ...
}:
rec {
  default = sim;
  sim = pkgs.callPackage ./sim.nix {
    inherit
      pkgs
      inputs
      lib
      self
      ;
  };
}
