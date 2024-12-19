# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 11:39:24 2024

@author: DURAND
"""

import xarray as xr
import numpy as np
from pathlib import Path
import os
import traceback
import shutil

def get_coordinates_info(ds):
    """
    Détermine le type de coordonnées et retourne les informations nécessaires
    """
    # Vérifier si nous avons des coordonnées rotated pole
    if 'rotated_pole' in ds.variables and 'rlat' in ds.dims and 'rlon' in ds.dims:
        return {
            'type': 'rotated_pole',
            'lat_name': 'rlat',
            'lon_name': 'rlon'
        }
    
    # Vérifier si nous avons des coordonnées projetées (x/y)
    if 'y' in ds.dims and 'x' in ds.dims:
        # Chercher les variables de latitude/longitude
        lat_var = None
        lon_var = None
        
        # Chercher les coordonnées 2D
        for var in ds.variables:
            if ds[var].dims == ('y', 'x'):
                if any(name in var.lower() for name in ['lat', 'latitude']):
                    lat_var = var
                elif any(name in var.lower() for name in ['lon', 'longitude']):
                    lon_var = var
        
        if lat_var and lon_var:
            return {
                'type': 'projected',
                'x_name': 'x',
                'y_name': 'y',
                'lat_var': lat_var,
                'lon_var': lon_var
            }
    
    # Vérifier les coordonnées lat/lon directes
    lat_candidates = [dim for dim in ds.dims if dim.lower() in ['lat', 'latitude']]
    lon_candidates = [dim for dim in ds.dims if dim.lower() in ['lon', 'longitude']]
    
    if lat_candidates and lon_candidates:
        return {
            'type': 'latlon',
            'lat_name': lat_candidates[0],
            'lon_name': lon_candidates[0]
        }
    
    raise ValueError(f"Système de coordonnées non reconnu. Variables disponibles: {list(ds.variables.keys())}")

def process_netcdf_file(input_file, mask_file, output_dir):
    """
    Traite un fichier NetCDF en appliquant le masque France et en exportant le résultat
    """
    input_file = Path(input_file)
    output_dir = Path(output_dir)
    
    # Vérification de sécurité - le fichier de sortie ne doit pas avoir le même chemin que l'entrée
    output_file = output_dir / f"france_{input_file.name}"
    if output_file == input_file:
        raise ValueError(f"Erreur de sécurité : Le fichier de sortie aurait le même chemin que l'entrée : {input_file}")
    
    print(f"\nTraitement de {input_file}")
    print(f"Fichier de sortie prévu : {output_file}")
    
    try:
        # Charger les datasets
        mask_ds = xr.open_dataset(mask_file, mode='r')
        ds = xr.open_dataset(input_file, mode='r')
        
        # Obtenir les informations sur les coordonnées
        coords_info = get_coordinates_info(ds)
        print(f"Type de coordonnées détecté: {coords_info['type']}")
        
        if coords_info['type'] == 'latlon':
            # Cas des coordonnées lat/lon classiques
            lat_name = coords_info['lat_name']
            lon_name = coords_info['lon_name']
            
            # Découper sur la France
            ds_cropped = ds.sel(
                {lat_name: slice(41, 51.5),
                 lon_name: slice(-5, 10)}
            )
            
            # Obtenir les grilles lat/lon pour l'interpolation
            lats = ds_cropped[lat_name]
            lons = ds_cropped[lon_name]
            
        else:  # coords_info['type'] == 'projected' ou 'rotated_pole'
            # Cas des coordonnées projetées
            lat_var = 'lat'
            lon_var = 'lon'
            lats = ds[lat_var]
            lons = ds[lon_var]
            
            # Créer un masque pour la France
            france_mask = (lats >= 41) & (lats <= 51.5) & (lons >= -5) & (lons <= 10)
            
            # Appliquer le masque géographique avec drop=True pour réduire la taille
            ds_cropped = ds.where(france_mask, drop=True)
            
            # Mettre à jour les lats/lons pour l'interpolation
            lats = ds_cropped[lat_var]
            lons = ds_cropped[lon_var]
        
        # Interpoler le masque sur la grille des données
        mask_interp = mask_ds.land_sea_mask.interp(
            latitude=lats,
            longitude=lons,
            method='nearest'
        )
        
        # Appliquer le masque à toutes les variables
        ds_masked = ds_cropped.copy()
        for var in ds_masked.data_vars:
            # Gestion spéciale des variables bounds
            if 'bounds' in var and 'nvertex' in ds_masked[var].dims:
                mask_expanded = np.expand_dims(mask_interp, axis=-1)
                mask_expanded = np.repeat(mask_expanded, ds_masked[var].sizes['nvertex'], axis=-1)
                ds_masked[var] = ds_masked[var].where(mask_expanded)
            # Pour les autres variables
            elif any(dim in ds_masked[var].dims for dim_set in [['y', 'x'], ['rlat', 'rlon']] 
                    for dim in dim_set):
                mask_expanded = mask_interp
                # Ajouter les dimensions temporelles si nécessaire
                for dim in ds_masked[var].dims:
                    if dim not in ['x', 'y', 'rlat', 'rlon']:
                        mask_expanded = np.expand_dims(mask_expanded, axis=0)
                ds_masked[var] = ds_masked[var].where(mask_expanded)
        
        # Créer le dossier de sortie si nécessaire
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder avec compression
        encoding = {var: {'zlib': True, 'complevel': 5} for var in ds_masked.data_vars}
        
        print("Sauvegarde du nouveau fichier...")
        ds_masked.to_netcdf(output_file, encoding=encoding)
        
        # Vérifier que le fichier d'origine existe toujours
        if not input_file.exists():
            raise RuntimeError(f"ERREUR CRITIQUE : Le fichier d'origine a disparu : {input_file}")
        
        # Afficher les statistiques de taille
        original_size = os.path.getsize(input_file)/1024/1024
        final_size = os.path.getsize(output_file)/1024/1024
        reduction = ((original_size - final_size) / original_size) * 100
        
        print(f"Fichier original intact : {input_file}")
        print(f"Fichier sauvegardé : {output_file}")
        print(f"Taille originale : {original_size:.1f} MB")
        print(f"Taille après traitement : {final_size:.1f} MB")
        print(f"Réduction : {reduction:.1f}%")
        
        return output_file
        
    except Exception as e:
        print(f"Erreur détaillée lors du traitement de {input_file}:")
        print(f"Variables disponibles: {list(ds.variables.keys())}")
        traceback.print_exc()
        if input_file.exists():
            print("Le fichier original est intact.")
        raise
        
    finally:
        # S'assurer que les datasets sont toujours fermés
        try:
            ds.close()
            ds_masked.close()
            mask_ds.close()
        except:
            pass

def create_output_directory_structure(base_input_dir, base_output_dir):
    """
    Crée la structure de dossiers de sortie identique à celle d'entrée
    """
    for input_dir in Path(base_input_dir).glob("*"):
        if input_dir.is_dir() and input_dir.name != 'france':
            output_dir = Path(base_output_dir) / input_dir.name
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"Création du dossier de sortie: {output_dir}")

def process_all_directories(base_input_dir, base_output_dir, mask_file):
    """
    Traite tous les fichiers NetCDF dans tous les sous-dossiers
    """
    # Créer d'abord la structure de dossiers
    create_output_directory_structure(base_input_dir, base_output_dir)
    
    # Compteurs pour les statistiques
    total_files = 0
    processed_files = 0
    failed_files = 0
    total_size_reduction = 0
    
    # Parcourir tous les dossiers
    for input_dir in Path(base_input_dir).glob("*"):
        if input_dir.is_dir() and input_dir.name != 'france':
            print(f"\nTraitement du dossier: {input_dir}")
            output_dir = Path(base_output_dir) / input_dir.name
            
            # Trouver tous les fichiers .nc dans le dossier
            nc_files = list(input_dir.glob("*.nc"))
            total_files += len(nc_files)
            
            # Traiter chaque fichier
            for i, file in enumerate(nc_files, 1):
                try:
                    print(f"\nTraitement du fichier {i}/{len(nc_files)} dans {input_dir.name}")
                    process_netcdf_file(file, mask_file, output_dir)
                    processed_files += 1
                    
                    # Calculer la réduction de taille
                    original_size = os.path.getsize(file)/1024/1024
                    output_file = output_dir / f"france_{file.name}"
                    final_size = os.path.getsize(output_file)/1024/1024
                    reduction = ((original_size - final_size) / original_size) * 100
                    total_size_reduction += reduction
                    
                except Exception as e:
                    print(f"Erreur lors du traitement de {file}: {str(e)}")
                    failed_files += 1
    
    # Afficher les statistiques finales
    print("\n=== Statistiques finales ===")
    print(f"Nombre total de fichiers traités: {total_files}")
    print(f"Fichiers traités avec succès: {processed_files}")
    print(f"Fichiers en erreur: {failed_files}")
    if processed_files > 0:
        avg_reduction = total_size_reduction / processed_files
        print(f"Réduction de taille moyenne: {avg_reduction:.1f}%")

if __name__ == "__main__":
    # Chemins à configurer
    mask_file = "france_land_sea_mask.nc"
    base_input_dir = "."  # Dossier courant
    base_output_dir = "./france"
    
    try:
        process_all_directories(base_input_dir, base_output_dir, mask_file)
        print("\nTraitement terminé!")
        
    except Exception as e:
        print(f"\nUne erreur s'est produite: {str(e)}")
        traceback.print_exc()