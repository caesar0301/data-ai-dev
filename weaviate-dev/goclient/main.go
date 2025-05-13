package main

import (
	"context"
	"fmt"
	"time"

	"github.com/golang/glog"
	"github.com/weaviate/weaviate-go-client/v5/weaviate"
	"github.com/weaviate/weaviate/entities/models"
)

const (
	WeaviateHTTPEndpoint = "localhost:32350"
	OllamaEndpoint       = "http://myrelease-ollama:11434"
)

func initWeaviateClient() (*weaviate.Client, error) {
	cfg := weaviate.Config{
		Host:       WeaviateHTTPEndpoint,
		Scheme:     "http",
		Headers:    nil,
		AuthConfig: nil,
	}

	client, err := weaviate.NewClient(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create weaviate driver: %v", err)
	}

	_, err = client.Misc().LiveChecker().Do(context.Background())
	if err != nil {
		return nil, fmt.Errorf("failed to verify weaviate connectivity: %v", err)
	}
	glog.Info("Weaviate connection established.")
	return client, nil
}

func initCollectionSchema(client *weaviate.Client) error {
	tenants := []models.Tenant{
		{
			Name: "admin",
		},
	}
	err := client.Schema().TenantsCreator().
		WithClassName("Droplet").
		WithTenants(tenants...).
		Do(context.Background())
	if err != nil {
		return fmt.Errorf("failed to create tenants: %v", err)
	}

	// Define class schema
	dropletClass := &models.Class{
		Class:       "Droplet",
		Description: "A collection representing droplets with metadata and location.",
		Properties: []*models.Property{
			{
				Name:     "uuid",
				DataType: []string{"text"},
			},
			{
				Name:     "createdTime",
				DataType: []string{"date"},
			},
			{
				Name:     "modifiedTime",
				DataType: []string{"date"},
			},
			{
				Name:     "geoCoordinates",
				DataType: []string{"geoCoordinates"},
			},
			{
				Name:     "text",
				DataType: []string{"text"},
			},
			{
				Name:     "text_format",
				DataType: []string{"text"},
			},
		},
		Vectorizer: "text2vec-ollama",
		ModuleConfig: map[string]any{
			"text2vec-ollama": map[string]any{
				"model":              "nomic-embed-text:latest",
				"vectorizeClassName": true,
				"apiEndpoint":        OllamaEndpoint,
			},
			"generative-ollama": map[string]any{
				"model":       "llama3.2:3b",
				"enabled":     true,
				"apiEndpoint": OllamaEndpoint,
			},
		},
		MultiTenancyConfig: &models.MultiTenancyConfig{
			Enabled: true,
		},
	}

	// Create schema
	client.Schema().ClassDeleter().WithClassName(dropletClass.Class).Do(context.Background())
	if err := client.Schema().ClassCreator().WithClass(dropletClass).Do(context.Background()); err != nil {
		return fmt.Errorf("failed to create class: %v", err)
	}
	fmt.Println("Class 'Droplet' created successfully.")
	return nil
}

func insertData(client *weaviate.Client) error {
	// Insert example Droplet object
	dropletProps := map[string]string{
		"uuid":         "abc123",
		"createdTime":  time.Now().Format(time.RFC3339),
		"modifiedTime": time.Now().Format(time.RFC3339),
		"text":         "A sample droplet of thought.",
		"text_format":  "markdown",
	}

	created, err := client.Data().Creator().
		WithClassName("Droplet").
		WithProperties(dropletProps).
		WithTenant("admin").
		Do(context.Background())
	if err != nil {
		return fmt.Errorf("failed to insert data: %v", err)
	}
	fmt.Printf("Example Droplet object inserted successfully: %v", created)
	return nil
}

func main() {
	client, err := initWeaviateClient()
	if err != nil {
		panic(err)
	} else {
		fmt.Println("Weaviate client initialized.")
	}
	if err := initCollectionSchema(client); err != nil {
		panic(err)
	} else {
		fmt.Println("Collection schema initialized.")
	}
	if err := insertData(client); err != nil {
		panic(err)
	} else {
		fmt.Println("Data inserted successfully.")
	}
}
