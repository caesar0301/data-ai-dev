package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"time"

	"github.com/golang/glog"
	"github.com/weaviate/weaviate-go-client/v5/weaviate"
	"github.com/weaviate/weaviate/entities/models"
)

const (
	WeaviateHTTPEndpoint = "localhost:31848"
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
	// Define class schema
	// Load schema from JSON file
	schemaFile, err := os.ReadFile("droplet_schema.json")
	if err != nil {
		panic(fmt.Errorf("failed to read schema file: %w", err))
	}

	var dropletClass models.Class
	if err := json.Unmarshal(schemaFile, &dropletClass); err != nil {
		panic(fmt.Errorf("failed to parse schema JSON: %w", err))
	}

	// Create schema
	if err := client.Schema().ClassDeleter().WithClassName(dropletClass.Class).Do(context.Background()); err != nil {
		return fmt.Errorf("failed to delete class: %v", err)
	}
	if err := client.Schema().ClassCreator().WithClass(&dropletClass).Do(context.Background()); err != nil {
		return fmt.Errorf("failed to create class: %v", err)
	}
	fmt.Println("Class 'Droplet' created successfully.")

	tenants := []models.Tenant{
		{
			Name: "admin",
		},
	}
	if err := client.Schema().TenantsCreator().
		WithClassName("Droplet").
		WithTenants(tenants...).
		Do(context.Background()); err != nil {
		return fmt.Errorf("failed to create tenants: %v", err)
	}
	fmt.Println("Tenant admin created successfully.")
	return nil
}

func insertData(client *weaviate.Client) error {
	// Insert example Droplet object
	dropletProps := map[string]any{
		"uuid":          "abc123",
		"created_time":  time.Now().UnixMilli(),
		"modified_time": time.Now().UnixMilli(),
		"text":          "A sample droplet of thought.",
		"text_format":   "markdown",
	}

	if created, err := client.Data().Creator().
		WithClassName("Droplet").
		WithProperties(dropletProps).
		WithTenant("admin").
		Do(context.Background()); err != nil {
		return fmt.Errorf("failed to insert data: %v", err)
	} else {
		fmt.Printf("Example Droplet object inserted successfully: %v\n", created.Object)
	}

	entries, err := client.Data().ObjectsGetter().
		WithClassName("Droplet").
		WithTenant("admin").
		Do(context.Background())
	if err != nil {
		return fmt.Errorf("failed to get objects: %v", err)
	}
	for _, entry := range entries {
		fmt.Printf("Entry: %v\n", entry.ID.String())
	}
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
